from __future__ import annotations

import functools
import inspect
import itertools
import logging
import operator
import sys
import typing
from collections import ChainMap
from collections import defaultdict
from collections import deque
from collections import namedtuple
from collections.abc import Callable
from collections.abc import Collection
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import MutableMapping
from collections.abc import Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from copy import copy
from dataclasses import _MISSING_TYPE
from dataclasses import KW_ONLY
from dataclasses import Field
from dataclasses import InitVar
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from itertools import tee as tee_iterable
from random import Random
from textwrap import dedent
from types import EllipsisType
from types import FrameType
from types import UnionType
from typing import Annotated
from typing import Any
from typing import ClassVar
from typing import Literal
from typing import NamedTuple
from typing import Protocol
from typing import Self
from typing import cast
from typing import dataclass_transform
from typing import overload
from weakref import ref

from docnote import ClcNote
from typing_extensions import TypeIs

from templatey._annotations import InterfaceAnnotation
from templatey._annotations import InterfaceAnnotationFlavor
from templatey.interpolators import NamedInterpolator
from templatey.parser import InterpolatedFunctionCall
from templatey.parser import InterpolationConfig
from templatey.parser import NestedContentReference
from templatey.parser import NestedVariableReference
from templatey.parser import ParsedTemplateResource

if typing.TYPE_CHECKING:
    from _typeshed import DataclassInstance
else:
    DataclassInstance = object

# Technically this should be an intersection type with both the
# _TemplateIntersectable from templates and the DataclassInstance returned by
# the dataclass transform. Unfortunately, type intersections don't yet exist in
# python, so we have to resort to this (overly broad) type
TemplateParamsInstance = DataclassInstance
type TemplateClass = type[TemplateParamsInstance]
logger = logging.getLogger(__name__)
# Defining this as a contextvar basically just for testing purposes. We want
# everything else to get the default global.
_PENDING_FORWARD_REFS: ContextVar[
    dict[ForwardRefLookupKey, set[TemplateClass]]] = ContextVar(
        '_PENDING_FORWARD_REFS', default=defaultdict(set))  # noqa: B039

# Note: we don't need cryptographically secure IDs here, so let's preserve
# entropy (might also be faster, dunno). Also note: the only reason we're
# using a contextvar here is so that we can theoretically replace it with
# a deterministic seed during testing (if we run into flakiness due to
# non-determinism)
_ID_PRNG: ContextVar[Random] = ContextVar('_ID_PRNG', default=Random())  # noqa: B039, S311
_ID_BITS = 128
# This is used by ``anchor_closure_scope`` to assign a scope ID to templates.
# It's used for all templates, but will only be non-null inside a closure.
_CURRENT_SCOPE_ID: ContextVar[int | None] = ContextVar(
    '_CURRENT_SCOPE_ID', default=None)


# Technically, these should use the TemplateIntersectable from templates.py,
# but since we can't define type intersections yet...
type Slot[T: TemplateParamsInstance] = Annotated[
    Sequence[T] | EllipsisType,
    InterfaceAnnotation(InterfaceAnnotationFlavor.SLOT)]
type Var[T] = Annotated[
    T | EllipsisType,
    InterfaceAnnotation(InterfaceAnnotationFlavor.VARIABLE)]
type Content[T] = Annotated[
    T,
    InterfaceAnnotation(InterfaceAnnotationFlavor.CONTENT)]


class TemplateIntersectable(Protocol):
    """This is the actual template protocol, which we would
    like to intersect with the TemplateParamsInstance, but cannot.
    Primarily included for documentation.
    """
    _templatey_config: ClassVar[TemplateConfig]
    # Note: whatever kind of object this is, it needs to be understood by the
    # template loader defined in the template environment. It would be nice for
    # this to be a typvar, but python doesn't currently support typevars in
    # classvars
    _templatey_resource_locator: ClassVar[object]
    _templatey_signature: ClassVar[TemplateSignature]
    # Oldschool here for performance reasons; otherwise this would be a dict.
    # Field names match the field names from the params; the value is gathered
    # from the metadata value on the field.
    _templatey_prerenderers: ClassVar[NamedTuple]


def is_template_class(cls: type) -> TypeIs[type[TemplateIntersectable]]:
    """Rather than relying upon @runtime_checkable, which doesn't work
    with protocols with ClassVars, we implement our own custom checker
    here for narrowing the type against TemplateIntersectable. Note
    that this also, I think, might be usable for some of the issues
    re: the missing intersection type in python, though support might be
    unreliable depending on which type checker is in use.
    """
    return (
        hasattr(cls, '_templatey_config')
        and hasattr(cls, '_templatey_resource_locator')
        and hasattr(cls, '_templatey_signature')
    )


def is_template_instance(instance: object) -> TypeIs[TemplateIntersectable]:
    """Rather than relying upon @runtime_checkable, which doesn't work
    with protocols with ClassVars, we implement our own custom checker
    here for narrowing the type against TemplateIntersectable. Note
    that this also, I think, might be usable for some of the issues
    re: the missing intersection type in python, though support might be
    unreliable depending on which type checker is in use.
    """
    return is_template_class(type(instance))


class VariableEscaper(Protocol):

    def __call__(self, value: str) -> str:
        """Variable escaper functions accept a single positional
        argument: the value of the variable to escape. It then does any
        required escaping and returns the final string.
        """
        ...


class ContentVerifier(Protocol):

    def __call__(self, value: str) -> Literal[True]:
        """Content verifier functions accept a single positional
        argument: the value of the content to verify. It does any
        verification, and then returns True if the content was okay,
        or raises BlockedContentValue if the content was not acceptable.

        Note that we raise instead of trying to escape for two reasons:
        1.. We don't really know what to replace it with. This is also
            true with variables, but:
        2.. We expect that content is coming from -- if not trusted,
            then at least authoritative -- sources, and therefore, we
            should fail loudly, because it gives the author a chance to
            correct the problem before it becomes user-facing.
        """
        ...


class InterpolationPrerenderer(Protocol):

    def __call__(
            self,
            value: Annotated[
                object | None,
                ClcNote(
                    '''The value of the variable or content. A value of
                    ``None`` indicates that the value is intended to be
                    omitted, but you may still provide a fallback
                    instead.
                    ''')]
            ) -> str | None:
        """Interpolation prerenderers give you a chance to modify the
        rendered result of a particular content or variable value, omit
        it entirely, or provide a fallback for missing values.

        Prerenderers are applied before formatting, escaping, and
        verification, and the result of the prerenderer is used to
        determine whether or not the value should be included in the
        result. If your prerenderer returns ``None``, the parameter will
        be completely omitted (including any prefix or suffix).
        """
        ...


# The following is adapted directly from typeshed. We did some formatting
# updates, and inserted our prerenderer param.
if sys.version_info >= (3, 14):
    @overload
    def param[_T](
            *,
            default: _T,
            default_factory: Literal[_MISSING_TYPE.MISSING] = ...,
            init: bool = True,
            repr: bool = True,
            hash: bool | None = None,
            compare: bool = True,
            metadata: Mapping[Any, Any] | None = None,
            kw_only: bool | Literal[_MISSING_TYPE.MISSING] = ...,
            doc: str | None = None,
            prerenderer: InterpolationPrerenderer | None = None,
            ) -> _T: ...
    @overload
    def param[_T](
            *,
            default: Literal[_MISSING_TYPE.MISSING] = ...,
            default_factory: Callable[[], _T],
            init: bool = True,
            repr: bool = True,
            hash: bool | None = None,
            compare: bool = True,
            metadata: Mapping[Any, Any] | None = None,
            kw_only: bool | Literal[_MISSING_TYPE.MISSING] = ...,
            doc: str | None = None,
            prerenderer: InterpolationPrerenderer | None = None,
            ) -> _T: ...
    @overload
    def param[_T](
            *,
            default: Literal[_MISSING_TYPE.MISSING] = ...,
            default_factory: Literal[_MISSING_TYPE.MISSING] = ...,
            init: bool = True,
            repr: bool = True,
            hash: bool | None = None,
            compare: bool = True,
            metadata: Mapping[Any, Any] | None = None,
            kw_only: bool | Literal[_MISSING_TYPE.MISSING] = ...,
            doc: str | None = None,
            prerenderer: InterpolationPrerenderer | None = None,
            ) -> Any: ...

# This is technically only valid for >=3.10, but we require that anyways
else:
    @overload
    def param[_T](
            *,
            default: _T,
            default_factory: Literal[_MISSING_TYPE.MISSING] = ...,
            init: bool = True,
            repr: bool = True,
            hash: bool | None = None,
            compare: bool = True,
            metadata: Mapping[Any, Any] | None = None,
            kw_only: bool | Literal[_MISSING_TYPE.MISSING] = ...,
            prerenderer: InterpolationPrerenderer | None = None,
            ) -> _T: ...
    @overload
    def param[_T](
            *,
            default: Literal[_MISSING_TYPE.MISSING] = ...,
            default_factory: Callable[[], _T],
            init: bool = True,
            repr: bool = True,
            hash: bool | None = None,
            compare: bool = True,
            metadata: Mapping[Any, Any] | None = None,
            kw_only: bool | Literal[_MISSING_TYPE.MISSING] = ...,
            prerenderer: InterpolationPrerenderer | None = None,
            ) -> _T: ...
    @overload
    def param[_T](
            *,
            default: Literal[_MISSING_TYPE.MISSING] = ...,
            default_factory: Literal[_MISSING_TYPE.MISSING] = ...,
            init: bool = True,
            repr: bool = True,
            hash: bool | None = None,
            compare: bool = True,
            metadata: Mapping[Any, Any] | None = None,
            kw_only: bool | Literal[_MISSING_TYPE.MISSING] = ...,
            prerenderer: InterpolationPrerenderer | None = None,
            ) -> Any: ...


def param(
        *,
        prerenderer: InterpolationPrerenderer | None = None,
        metadata: Mapping[Any, Any] | None = None,
        **field_kwargs):
    if metadata is None:
        metadata = {'templatey.prerenderer': prerenderer}

    else:
        metadata = {
            **metadata,
            'templatey.prerenderer': prerenderer}

    return field(**field_kwargs, metadata=metadata)


@dataclass_transform(field_specifiers=(param, field, Field))
def template[T: type](  # noqa: PLR0913
        config: TemplateConfig,
        template_resource_locator: object,
        /, *,
        init: bool = True,
        repr: bool = True,  # noqa: A002
        eq: bool = True,
        order: bool = False,
        unsafe_hash: bool = False,
        frozen: bool = False,
        match_args: bool = True,
        kw_only: bool = False,
        slots: bool = True,
        weakref_slot: bool = False
        ) -> Callable[[T], T]:
    """This both transforms the decorated class into a stdlib dataclass
    and declares it as a templatey template.

    **Note that unlike the stdlib dataclass decorator, this defaults to
    ``slots=True``.** If you find yourself having problems with
    metaclasses and/or subclassing, you can disable this by passing
    ``slots=False``. Generally speaking, though, this provides a
    free performance benefit. **If weakref support is required, be sure
    to pass ``weakref_slot=True``.
    """
    return functools.partial(
        make_template_definition,
        dataclass_kwargs={
            'init': init,
            'repr': repr,
            'eq': eq,
            'order': order,
            'unsafe_hash': unsafe_hash,
            'frozen': frozen,
            'match_args': match_args,
            'kw_only': kw_only,
            'slots': slots,
            'weakref_slot': weakref_slot
        },
        template_resource_locator=template_resource_locator,
        template_config=config)


@dataclass(frozen=True)
class TemplateConfig[T: type, L: object]:
    interpolator: Annotated[
        NamedInterpolator,
        ClcNote(
            '''The interpolator determines what characters are used for
            performing interpolations within the template. They can be
            escaped by repeating them, for example ``{{}}`` would be
            a literal ``{}`` with a curly braces interpolator.
            ''')]
    variable_escaper: Annotated[
        VariableEscaper,
        ClcNote(
            '''Variables are always escaped. The variable escaper is
            the callable responsible for performing that escaping. If you
            don't need escaping, there are noop escapers within the prebaked
            template configs that you can use for convenience.
            ''')]
    content_verifier: Annotated[
        ContentVerifier,
        ClcNote(
            '''Content isn't escaped, but it ^^is^^ verified. Content
            verification is a simple process that either succeeds or fails;
            it allows, for example, to allowlist certain HTML tags.
            ''')]


@dataclass(slots=True, frozen=True)
class TemplateProvenanceNode:
    """TemplateProvenanceNode instances are unique to the exact location
    on the exact render tree of a particular template instance. If the
    template instance gets reused within the same render tree, it will
    have multiple provenance nodes. And each different slot in an
    enclosing template will have a separate provenance node, potentially
    with different namespace overrides.

    This is used both during function execution (to calculate the
    concrete value of any parameters), and also while flattening the
    render tree.

    Also note that the root template, **along with any templates
    injected into the render by environment functions,** will have an
    empty list of provenance nodes.

    Note that, since overrides from the enclosing template can come
    exclusively from the template body -- and are therefore shared
    across all nested children of the same slot -- they don't get stored
    within the provenance, since we'd require access to the template
    bodies, which we don't yet have.
    """
    encloser_slot_key: str
    encloser_slot_index: int
    # The reason to have both the instance and the instance ID is so that we
    # can have hashability of the ID while not imposing an API on the instances
    instance_id: TemplateInstanceID
    instance: TemplateParamsInstance = field(compare=False)


class TemplateProvenance(tuple[TemplateProvenanceNode]):

    def bind_content(
            self,
            name: str,
            template_preload: dict[TemplateClass, ParsedTemplateResource]
            ) -> object:
        """Use this to calculate a concrete value for use in rendering.
        This walks up the provenance stack, recursively looking for any
        overrides to the content. If none are found, it returns the
        value from the childmost instance in the provenance.
        """
        # We use the literal ellipsis type as a sentinel for values not being
        # added yet, so we might as well just continue the trend!
        current_provenance_node = self[-1]
        value = getattr(current_provenance_node.instance, name, ...)
        encloser_param_name = name
        encloser_slot_key = current_provenance_node.encloser_slot_key
        for encloser in reversed(self[0:-1]):
            template_class = type(encloser.instance)
            encloser_template = template_preload[template_class]
            encloser_overrides = (
                encloser_template.slots[encloser_slot_key].params)

            if encloser_param_name in encloser_overrides:
                value = encloser_overrides[encloser_param_name]

                if isinstance(value, NestedContentReference):
                    encloser_slot_key = encloser.encloser_slot_key
                    encloser_param_name = value.name
                    value = ...
                else:
                    break
            else:
                break

        if value is ...:
            raise KeyError(
                'No value found for content with name at slot!',
                self[-1].instance, name)

        return value

    def bind_variable(
            self,
            name: str,
            template_preload: dict[TemplateClass, ParsedTemplateResource]
            ) -> object:
        """Use this to calculate a concrete value for use in rendering.
        This walks up the provenance stack, recursively looking for any
        overrides to the variable. If none are found, it returns the
        value from the childmost instance in the provenance.
        """
        # We use the literal ellipsis type as a sentinel for values not being
        # added yet, so we might as well just continue the trend!
        current_provenance_node = self[-1]
        value = getattr(current_provenance_node.instance, name, ...)
        encloser_param_name = name
        encloser_slot_key = current_provenance_node.encloser_slot_key
        for encloser in reversed(self[0:-1]):
            template_class = type(encloser.instance)
            encloser_template = template_preload[template_class]
            encloser_overrides = (
                encloser_template.slots[encloser_slot_key].params)

            if encloser_param_name in encloser_overrides:
                value = encloser_overrides[encloser_param_name]

                if isinstance(value, NestedVariableReference):
                    encloser_slot_key = encloser.encloser_slot_key
                    encloser_param_name = value.name
                    value = ...
                else:
                    break
            else:
                break

        if value is ...:
            raise KeyError(
                'No value found for variable with name at slot!',
                self[-1].instance, name)

        return value


@contextmanager
def anchor_closure_scope():
    """We strongly recommend against defining templates within a
    closure, as it can cause a number of fragility issues, and just
    generally makes less sense than defining templates at the module
    level. However, if you absolutely must create a new template within
    a closure, you must use ``anchor_closure_scope`` to give the
    templates a known closure scope. Can be used either as a decorator
    or a context:

    > Decorator usage
    __embed__: 'code/python'
        @anchor_closure_scope()
        def my_func():
            # template definition goes here
            ...

    > Context manager usage
    __embed__: 'code/python'
        def my_other_func():
            with anchor_closure_scope():
                # template definition goes here
                ...
    """
    token = _CURRENT_SCOPE_ID.set(_create_templatey_id())
    try:
        yield
    finally:
        _CURRENT_SCOPE_ID.reset(token)


def _create_templatey_id() -> int:
    """Templatey IDs are unique identifiers (theoretically, absent
    birthday collisions) that we currently use in two places:
    ++  as a scope ID, which is used when defining templates in closures
    ++  for giving slot tree nodes a unique reference target for
        recursion loops while copying and merging, which is more robust
        than ``id(target)`` and can be transferred via dataclass field
        into cloned/copied/merged nodes.
    """
    prng = _ID_PRNG.get()
    return prng.getrandbits(_ID_BITS)


@dataclass(frozen=True, slots=True)
class ForwardRefLookupKey:
    """We use this to find all possible ForwardRefLookupKey instances
    for a particular pending template.

    Note that, by definition, forward references can only happen in two
    situations:
    ++  within the same module or closure, by something declared later
        on during execution
    ++  because of something hidden behind a ``if typing.TYPE_CHECKING``
        block in imports
    (Any other scenario would result in import failures preventing the
    module's execution).

    Names imported behind ``TYPE_CHECKING`` blocks can **only** be
    resolved using explicit helpers in the ``@template`` decorator.
    (TODO: need to add those!). There's just no way around that one;
    by definition, it's a circular import, and the name isn't available
    at runtime. So you need an escape hatch.

    Therefore, unless passed in an explicit module name because of the
    aforementioned escape hatch, these must always happen from within
    the same module as the template itself.

    Furthermore, we make one assumption here for the purposes of
    doing as much work as possible at import time, ahead of the first
    call to render a template: that the enclosing template references
    the nested template by the nested template's proper name, and
    doesn't rename it.

    The only workaround for a renamed nested template would be to
    create a dedicated resolution function, to be called at first
    render time, that re-inspects the template's type annotations, and
    figures out exactly what type it uses at that point in time. That's
    a mess, so.... we'll punt on it.
    """
    module: str
    name: str
    scope_id: int | None


# Note: the ordering here is to emphasize the fact that the slot
# name is on the ENCLOSING template, but the slot type is from the
# NESTED template
class _SlotTreeRoute[T: _SlotTreeNode](tuple[str, TemplateClass, T]):
    """An individual route on the slot tree is defined by the attribute
    name for the slot, the slot type, and the subtree from the slot
    class.

    These are optimized for the non-union case. Traversing the slot tree
    with union types will result in a bunch of unnecessary comparisons
    against slot names of different slot types.

    Note that slot tree routes always have a concrete slot name and slot
    type, regardless of whether they're in a pending or concrete tree.
    The reason is simple: in a pending tree, all of the pending classes
    are dead-end nodes, and define their insertion points using just
    the string of the slot name, and nothing else.
    """
    @classmethod
    def new(
            cls,
            slot_name: str,
            slot_type: TemplateClass,
            subtree: T,
            ) -> _SlotTreeRoute[T]:
        return cls((slot_name, slot_type, subtree))

    @property
    def subtree(self) -> T:
        """This is slower than directly accessing the tuple values, but
        it makes for clearer code during tree building, where
        performance isn't quite so critical.
        """
        return self[2]

    @property
    def slot_path(self) -> tuple[str, TemplateClass]:
        return self[0:2]


@dataclass(slots=True)
class _SlotTreeNode[T: _SlotTreeNode](list[_SlotTreeRoute[T]]):
    """The purpose of the slot tree is to precalculate what sequences of
    getattr() calls we need to traverse to arrive at every instance of a
    particular slot type for a given template, including all nested
    templates.

    **These are optimized for rendering, not for template declaration.**
    Also note that these are optimized for slots that are not declared
    as type unions; type unions will result in a number of unnecessary
    comparisons against the routes of the other slot types in the union.

    The reason this is useful is for batching during rendering. This is
    important for function calls: it allows us to pre-execute all env
    func calls for a template before we start rendering it. In the
    future, it will also serve the same role for discovering the actual
    template types for dynamic slots, allowing us to load the needed
    template types in advance.

    An individual node on the slot tree is a list of all possible
    attribute names (as ``_SlotTreeRoute``s) that a particular search
    pass needs to check for a given instance. Note that **all** of the
    attributes must be searched -- hence using an iteration-optimized
    list instead of a mapping.
    """
    routes: InitVar[Iterable[_SlotTreeRoute[T]] | None] = None

    _: KW_ONLY
    # We use this to limit the number of entries we need in the transmogrifier
    # lookup during tree merging/copying
    is_recursive: bool = False
    # We use this to differentiate between dissimilar unions, one which
    # continues on to a subtree, and one which ends here with the target
    # instance
    is_terminus: bool = False

    id_: int = field(default_factory=_create_templatey_id)

    # We use this to make the logic cleaner when merging trees, but we want
    # the faster performance of the tuple when actually traversing the tree
    _routes_by_slot_path: \
        dict[tuple[str, TemplateClass], _SlotTreeRoute[T]] = field(init=False)

    def __post_init__(
            self,
            routes: Iterable[_SlotTreeRoute[T]] | None):
        # Explicit instead of super because... idunno, we're breaking things
        # somehow
        if routes is None:
            list.__init__(self)
        else:
            list.__init__(self, routes)

        self._routes_by_slot_path = {
            route.slot_path: route for route in self}

    def empty_clone(self) -> Self:
        """This creates a clone of the node without any routes. Useful
        for merging and copying, where you need to do some manual
        transform of the content.

        Note that this is almost the same as dataclasses.replace, with
        the exception that we create shallow copies of attributes
        instead of preserving them.
        """
        kwargs = {}
        for dc_field in fields(self):
            if dc_field.init:
                # Note: the copy here is important for any mutable values,
                # notably the insertion_slot_names.
                kwargs[dc_field.name] = copy(getattr(self, dc_field.name))

        return type(self)(**kwargs)

    def merge_fields_only(self, other: _SlotTreeNode):
        """Updates the current node, merging in all non-init field
        values from other, using |=. Only merges values that exist on
        the current node, allowing for transformation between pending
        and concrete node types.
        """
        missing = object()

        for dc_field in fields(self):
            if dc_field.init:
                current_value = getattr(self, dc_field.name)
                other_value = getattr(self, dc_field.name, missing)

                if other_value is not missing:
                    setattr(
                        self,
                        dc_field.name,
                        operator.ior(current_value, other_value))

    @property
    def requires_transmogrification(self) -> bool:
        """This determines whether or not copies of the tree require
        some post-processing to make sure that the tree STRUCTURE is
        the same. It is used by both copying and merging trees to make
        sure that the transmogrification lookup is as sparse as
        possible.

        For the base class, we simply wrap ``is_recursive``, but for
        the pending tree derived class, we also check other stuff --
        hence the wrapping.
        """
        return self.is_recursive

    def append(self, route: _SlotTreeRoute[T]):
        slot_path = (route[0], route[1])
        if slot_path in self._routes_by_slot_path:
            raise ValueError(
                'Templatey internal error: attempt to append duplicate slot '
                + 'name for same slot type! Please search for / report issue '
                + 'to github along with a traceback.')

        list.append(self, route)
        self._routes_by_slot_path[slot_path] = route

    def extend(self, routes: Iterable[_SlotTreeRoute[T]]):
        routes, routes_copy_1, routes_copy_2 = tee_iterable(routes, 3)

        # The slightly awkward sequencing here is so that it's an atomic
        # operation: check everything first, then update.
        # Not strictly necessary, but nice in the "be kind, rewind" kind of
        # way
        if any(
            (route_copy[0], route_copy[1]) in self._routes_by_slot_path
            for route_copy in routes_copy_1
        ):
            raise ValueError(
                'Templatey internal error: attempt to append duplicate slot '
                + 'name! Please search for / report issue to github along '
                + 'with a traceback.')

        list.extend(self, routes)
        self._routes_by_slot_path.update({
            (route_copy[0], route_copy[1]): route_copy
            for route_copy in routes_copy_2})

    def has_route_for(
            self,
            slot_name: str,
            slot_type: TemplateClass
            ) -> bool:
        return (slot_name, slot_type) in self._routes_by_slot_path

    def get_route_for(
            self,
            slot_name: str,
            slot_type: TemplateClass
            ) -> _SlotTreeRoute[T]:
        return self._routes_by_slot_path[(slot_name, slot_type)]


type _ConcreteSlotTreeNode = _SlotTreeNode[_ConcreteSlotTreeNode]


@dataclass(kw_only=True, slots=True)
class _PendingSlotTreeNode(_SlotTreeNode['_PendingSlotTreeNode']):
    # Note: the str is the slot_name that the route needs to be inserted under
    insertion_slot_names: set[str] = field(default_factory=set)

    @property
    def is_insertion_point(self) -> bool:
        return bool(self.insertion_slot_names)

    @property
    def requires_transmogrification(self) -> bool:
        return self.is_recursive


type TemplateInstanceID = int
type GroupedTemplateInvocations = dict[TemplateClass, list[TemplateProvenance]]
type TemplateLookupByID = dict[TemplateInstanceID, TemplateParamsInstance]
# Note that there's no need for an abstract version of this, at least not right
# now, because in order to calculate it, we'd need to know the template body,
# which doesn't happen until we already know the template instance, which means
# we can skip ahead to the concrete version.
type EnvFuncInvocation = tuple[TemplateProvenance, InterpolatedFunctionCall]


@dataclass(slots=True)
class TemplateSignature:
    """This class stores the processed interface based on the params.
    It gets used to compare with the TemplateParse to make sure that
    the two can be used together.

    Not meant to be created directly; instead, you should use the
    TemplateSignature.new() convenience method.
    """
    # It's nice to have this available, especially when resolving forward refs,
    # but unlike eg the slot tree, it's trivially easy for us to avoid GC
    # loops within the signature
    template_cls_ref: ref[TemplateClass]
    _forward_ref_lookup_key: ForwardRefLookupKey

    slot_names: frozenset[str]
    var_names: frozenset[str]
    content_names: frozenset[str]

    # Note that these contain all included types, not just the ones on the
    # outermost layer that are associated with the signature. In other words,
    # they include the flattened recursion of all included slots, all the way
    # down the tree
    _slot_tree_lookup: dict[TemplateClass, _ConcreteSlotTreeNode]
    _pending_ref_lookup: dict[ForwardRefLookupKey, _PendingSlotTreeContainer]

    # I really don't like that we need to remember to recalculate this every
    # time we update the slot tree lookup, but for rendering performance
    # reasons we want this to be precalculated before every call to render.
    included_template_classes: frozenset[TemplateClass] = field(init=False)

    def __post_init__(self):
        self.refresh_included_template_classes_snapshot()
        self.refresh_pending_forward_ref_registration()

    def refresh_included_template_classes_snapshot(self):
        """Call this when resolving forward references to apply any
        changes made to the slot tree to the template classes snapshot
        we use for increased render performance.
        """
        template_cls = self.template_cls_ref()
        if template_cls is None:
            raise RuntimeError(
                'Template class was garbage collected before template '
                + 'signature, and then signature asked to refresh included '
                + 'classes snapshot?!')

        self.included_template_classes = frozenset(
            {template_cls, *self._slot_tree_lookup})

    def refresh_pending_forward_ref_registration(self):
        """Call this after having resolved forward references (or when
        initially constructing the template signature) to register the
        template class as requiring its forward refs. This is what
        plumbs up the notification code to actually initiate resolving.
        """
        template_cls = self.template_cls_ref()
        if template_cls is None:
            raise RuntimeError(
                'Template class was garbage collected before template '
                + 'signature, and then signature asked to refresh pending '
                + 'forward ref registration?!')

        forward_ref_registry = _PENDING_FORWARD_REFS.get()
        for forward_ref in self._pending_ref_lookup:
            forward_ref_registry[forward_ref].add(template_cls)

    @classmethod
    def new(
            cls,
            template_cls: type,
            slots: dict[str,
                TemplateClass
                | UnionType
                | type[ForwardReferenceProxyClass]],
            vars_: dict[str, type | type[ForwardReferenceProxyClass]],
            content: dict[str, type | type[ForwardReferenceProxyClass]],
            *,
            forward_ref_lookup_key: ForwardRefLookupKey
            ) -> TemplateSignature:
        """Create a new TemplateSignature based on the gathered slots,
        vars, and content. This does all of the convenience calculations
        needed to populate the semi-redundant fields.
        """
        slot_names = frozenset(slots)
        var_names = frozenset(vars_)
        content_names = frozenset(content)

        # Quick refresher: our goal here is to construct a lookup that gets
        # us a route to every instance of a particular template type. In other
        # words, we want to be able to check a template type, and then see all
        # possible getattr() sequences that arrive at an instance of that
        # template type.
        tree_wip: dict[TemplateClass, _ConcreteSlotTreeNode]
        tree_wip = defaultdict(_SlotTreeNode)
        pending_ref_lookup: \
            dict[ForwardRefLookupKey, _PendingSlotTreeContainer] = {}
        for slot_name, slot_annotation in slots.items():
            cls._extend_wip_slot_and_ref_trees(
                template_cls,
                forward_ref_lookup_key,
                slot_name,
                slot_annotation,
                slot_tree_lookup=tree_wip,
                pending_ref_lookup=pending_ref_lookup)
        tree_wip.default_factory = None

        return cls(
            _forward_ref_lookup_key=forward_ref_lookup_key,
            template_cls_ref=ref(template_cls),
            slot_names=slot_names,
            var_names=var_names,
            content_names=content_names,
            _slot_tree_lookup=tree_wip,
            _pending_ref_lookup=pending_ref_lookup)

    @classmethod
    def _extend_wip_slot_and_ref_trees(
            cls,
            # This is used as a recursion guard. If we have a simple recursion,
            # there (somewhat surprisingly) isn't a forward ref at all when
            # getting the type hints, but instead, the actual class. But we're
            # still in the middle of populating its xable attributes, so we
            # need to short circuit it.
            template_cls: type,
            template_forward_ref_lookup_key: ForwardRefLookupKey,
            slot_name: str,
            slot_annotation:
                TemplateClass
                | UnionType
                | type[ForwardReferenceProxyClass],
            *,
            slot_tree_lookup:
                defaultdict[TemplateClass, _ConcreteSlotTreeNode],
            pending_ref_lookup:
                dict[ForwardRefLookupKey, _PendingSlotTreeContainer]
            ) -> None:
        """Builds the slot tree for a single slot on a template class.

        Keep in mind that child slots might themselves include pending
        trees, so we can't infer based on the annotation type whether
        or not the result will include them or not.
        """
        slot_types: Collection[
            TemplateClass
            | type[ForwardReferenceProxyClass]]

        if isinstance(slot_annotation, UnionType):
            slot_types = slot_annotation.__args__
        else:
            slot_types = (slot_annotation,)

        for slot_type in slot_types:
            if is_forward_reference_proxy(slot_type):
                forward_ref = slot_type.REFERENCE_TARGET

                existing_pending_tree = pending_ref_lookup.get(forward_ref)
                if existing_pending_tree is None:
                    dest_insertion = _PendingSlotTreeNode(
                        insertion_slot_names={slot_name})
                    pending_ref_lookup[forward_ref] = (
                        _PendingSlotTreeContainer(
                            pending_slot_type=forward_ref,
                            pending_root_node=dest_insertion))

                else:
                    (
                        existing_pending_tree
                        .pending_root_node
                        .insertion_slot_names.add(slot_name))

            # In the simple recursion case -- a template defines a slot of its
            # own class -- we can immediately create a reference loop without
            # needing a forward ref.
            elif slot_type is template_cls:
                recursive_slot_tree = slot_tree_lookup[slot_type]
                recursive_slot_tree.is_recursive = True
                recursive_slot_route = _SlotTreeRoute.new(
                    slot_name,
                    slot_type,
                    recursive_slot_tree)
                recursive_slot_tree.append(recursive_slot_route)

            # Remember that we expanded the union already, so this is
            # guaranteed to be a single concrete ``slot_type``.
            else:
                cls._extend_wip_slot_and_ref_trees_for_concrete_slot_type(
                    template_cls,
                    slot_name,
                    slot_type,
                    template_forward_ref_lookup_key,
                    slot_tree_lookup=slot_tree_lookup,
                    pending_ref_lookup=pending_ref_lookup)

    @staticmethod
    def _extend_wip_slot_and_ref_trees_for_concrete_slot_type(
            template_cls: type,
            slot_name: str,
            slot_type: TemplateClass,
            encloser_forward_ref_lookup_key: ForwardRefLookupKey,
            *,
            slot_tree_lookup:
                defaultdict[TemplateClass, _ConcreteSlotTreeNode],
            pending_ref_lookup:
                dict[ForwardRefLookupKey, _PendingSlotTreeContainer]
            ) -> None:
        """This carves out the slot tree extension for concrete slot
        types into a dedicated helper function to (theoretically) make
        testing easier. I say theoretically, because we don't yet have
        any unit tests for this part of the code, which could make use
        of the better organization.
        """
        enclosing_slot_tree = slot_tree_lookup[slot_type]
        # Keep in mind that we're mapping out all branches of the tree
        # here. Templates defining recursive loops will always have
        # both a terminus and a subtree. So we don't want to overwrite
        # anything that's already there; we simply want to note that
        # it can also have a terminus.
        if enclosing_slot_tree.has_route_for(slot_name, slot_type):
            existing_route = enclosing_slot_tree.get_route_for(
                slot_name, slot_type)
            existing_route.subtree.is_terminus = True

        else:
            enclosing_slot_tree.append(
                _SlotTreeRoute.new(
                    slot_name,
                    slot_type,
                    _SlotTreeNode(is_terminus=True)))

        slot_xable = cast(
            type[TemplateIntersectable], slot_type)
        nested_lookup = (
            slot_xable._templatey_signature._slot_tree_lookup)
        nested_pending_refs = (
            slot_xable._templatey_signature._pending_ref_lookup)

        # Note that because of the nested for loop, this will put all
        # possible slots from the entire union on equal footing.
        for (
            nested_slot_type, nested_slot_tree
        ) in nested_lookup.items():
            _merge_into_slot_tree(
                slot_type,
                slot_tree_lookup[nested_slot_type],
                nested_slot_tree)

        # Okay, now all we have left to do is transform all of the
        # pending references on the nested template into pending references on
        # the enclosing template.
        for (
            nested_forward_ref_key, nested_pending_slot_tree
        ) in nested_pending_refs.items():
            # Remember that we're in the middle of constructing the signature
            # for a new template class. If the nested class (from the slot) was
            # depending on the class we're still constructing, it hasn't yet
            # been updated with the resolved class. Therefore, instead of
            # needing to come back and fix up any recursive forward refs later,
            # we can simply do them right here, right now.
            # Also note that we'll NEVER have an existing pending tree for
            # this, because we're adding it directly and immediately into
            # the actual slot tree.
            if nested_forward_ref_key == encloser_forward_ref_lookup_key:
                _resolve_pending_slot_tree(
                    nested_pending_slot_tree,
                    slot_name,
                    enclosing_cls=template_cls,
                    enclosing_slot_tree_lookup=slot_tree_lookup,
                    enclosing_pending_ref_lookup=pending_ref_lookup,
                    resolved_cls=template_cls,
                    resolved_slot_tree_lookup=slot_tree_lookup,
                    resolved_pending_ref_lookup=pending_ref_lookup)
                continue

            # Remember that we're simply transforming the existing pending ref
            # tree from the nested slot into a pending ref tree on the
            # enclosing slot -- ie, we're not adding any additional insertion
            # points.
            existing_pending_tree = pending_ref_lookup.get(
                nested_forward_ref_key)

            if existing_pending_tree is None:
                # Transmogrified nodes gives us a lookup from the old
                # IDs to the new insertion nodes, allowing us to
                # convert the references to the copy.
                copied_tree = _copy_slot_tree(
                    nested_pending_slot_tree.pending_root_node)
                pending_ref_lookup[nested_forward_ref_key] = (
                    # Remember: we already expanded the slot type
                    # from unions!
                    _PendingSlotTreeContainer(
                        # Note: the additional tree layer here is because this
                        # is a slot on the enclosing template -- we need to
                        # descend a level!
                        pending_root_node=_PendingSlotTreeNode(
                            [_SlotTreeRoute.new(
                                slot_name,
                                # Note: this is the slot type for the CONCRETE
                                # class we're currently handling, NOT the
                                # pending one!
                                slot_type,
                                copied_tree)]),
                        pending_slot_type=nested_forward_ref_key))

            else:
                _merge_into_slot_tree(
                    slot_type,
                    existing_tree=existing_pending_tree.pending_root_node,
                    to_merge=nested_pending_slot_tree.pending_root_node)

    def extract_function_invocations(
            self,
            root_template_instance: TemplateParamsInstance,
            template_preload: dict[TemplateClass, ParsedTemplateResource]
            ) -> list[EnvFuncInvocation]:
        """Looks at all included abstract function invocations, and
        generates lists of their concrete invocations, based on both the
        actual values of slots at the template instances, as well as the
        template definition provided in template_preload.
        """
        invocations: list[EnvFuncInvocation] = []

        # Things to keep in mind when reading the following code:
        # ++  it may be easiest to step through using an example, perhaps from
        #     the test suite.
        # ++  enclosing template classes can have multiple slots with the same
        #     nested template class. We call these "parallel slots" in this
        #     function; they're places where the slot search tree needs to
        #     split into multiple branches
        # ++  the combinatorics here can be confusing, because we can have both
        #     multiple instances in each slot AND multiple slots for each
        #     template class
        # ++  multiple instances per slot branch the instance search tree, but
        #     multiple slots per template class branch the slot search tree.
        #     However, we need to exhaust both search trees when finding all
        #     relevant provenances. Therefore, we have to periodically refresh
        #     one of the search trees, whenever we step to a sibling on the
        #     other tree

        # The goal of the parallel slot backlog is to keep track of which other
        # SLOTS (attributes! not instances!) at a particular instance are ALSO
        # on the search path, and therefore need to be returned to after the
        # current branch has been exhausted.
        # The parallel slot backlog is a stack of stacks. The outer stack
        # corresponds to the depth in the slot tree. The inner stack
        # contains all remaining slots to search for a particular depth.
        parallel_slot_backlog_stack: list[list[_SlotTreeRoute]]

        # The goal of the instance backlog stack is to keep track of all the
        # INSTANCES (not slots / attributes!) that are also on the search path,
        # and therefore need to be returned to after the current branch has
        # been exhausted. Its deepest level gets refreshed from the instance
        # history stack every time we move on to a new parallel slot.
        # Similar (but different) to the parallel slot backlog, the instance
        # backlog is a stack of **queues** (it's important to preserve order,
        # since slot members are by definition ordered). The outer stack
        # again corresponds to the depth in the slot tree, and the inner queue
        # contains all of the remaining instances to search for a particular
        # depth.
        instance_backlog_stack: list[deque[TemplateProvenanceNode]]

        # The instance history stack is similar to the backlog stack; however,
        # it is not mutated on a particular level. Use it to "refresh" the
        # instance backlog stack for parallel slots.
        instance_history_stack: list[tuple[TemplateProvenanceNode, ...]]

        # These are all used per-iteration, and don't keep state across
        # iterations.
        nested_slot_name: str
        nested_slot_routes: _ConcreteSlotTreeNode
        nested_instances: Sequence[TemplateParamsInstance]

        # This is used in multiple loop iterations, plus at the end to add any
        # function calls for the root instance.
        root_provenance_node = TemplateProvenanceNode(
            encloser_slot_key='',
            encloser_slot_index=-1,
            instance_id=id(root_template_instance),
            instance=root_template_instance)

        # Keep in mind that the slot tree contains all included slot classes
        # (recursively), not just the ones at the root_template_instance.
        # Our goal here is:
        # 1.. find all template classes with abstract function calls
        # 2.. build provenances for all invocations of those template classes
        # 3.. combine (product) those provenances with all of the abstract
        #     function calls at that template class
        for template_class, root_nodes in self._slot_tree_lookup.items():
            abstract_calls = template_preload[template_class].function_calls

            # Constructing a provenance is relatively expensive, so we only
            # want to do it if we actually have some function calls within the
            # template
            if abstract_calls:
                provenances: list[TemplateProvenance] = []
                parallel_slot_backlog_stack = [list(root_nodes)]
                instance_history_stack = [(root_provenance_node,)]
                instance_backlog_stack = [deque(instance_history_stack[0])]

                # Our overall strategy here is to let the instance stack be
                # the primary driver. Only mutate other stacks when we're done
                # with a particular level in the instance backlog!
                while instance_backlog_stack:
                    # If there's nothing left on the current level of the
                    # instance backlog, there are a couple options.
                    # ++  We may have exhausted a particular parallel path
                    #     from the current level, but there are more left. We
                    #     need to refresh the list of instances and continue.
                    # ++  We may have exhausted all subtrees of the current
                    #     level. In that case, we need to back out a level and
                    #     continue looking for parallels, one level up.
                    if not instance_backlog_stack[-1]:
                        # Note that by checking for >1, we don't allocate a
                        # bunch of instance backlog children for nothing.
                        if len(parallel_slot_backlog_stack[-1]) > 1:
                            parallel_slot_backlog_stack[-1].pop()
                            instance_backlog_stack[-1].extend(
                                instance_history_stack[-1])

                        else:
                            parallel_slot_backlog_stack.pop()
                            instance_backlog_stack.pop()
                            instance_history_stack.pop()

                    # There are one or more remaining parallel paths from the
                    # current instances that lead to the target template_class.
                    # Choose the last one so we can pop it efficiently.
                    else:
                        (
                            nested_slot_name,
                            nested_slot_type,
                            nested_slot_routes
                        ) = parallel_slot_backlog_stack[-1][-1]

                        current_instance = (
                            instance_backlog_stack[-1][0].instance)
                        nested_instances = getattr(
                            current_instance, nested_slot_name)
                        nested_index = itertools.count()

                        # The parallel path we chose is a leaf node,
                        # which means that each nested instance is a
                        # provenance. Note that, because of recursion loops,
                        # this can happen whether or not there are nested
                        # slot routes, so we'll still check for those in just
                        # a second.
                        if nested_slot_routes.is_terminus:
                            partial_provenance = tuple(
                                instance_backlog_level[0]
                                for instance_backlog_level
                                in instance_backlog_stack)

                            # Note that using extend here is basically just a
                            # shorthand for repeatedly iterating on the
                            # outermost while loop after appending the
                            # children (like we did with nested_slot_routes)
                            provenances.extend(TemplateProvenance(
                                (
                                    *partial_provenance,
                                    TemplateProvenanceNode(
                                        encloser_slot_key=nested_slot_name,
                                        encloser_slot_index=next(nested_index),
                                        instance_id=id(nested_instance),
                                        instance=nested_instance)))
                                for nested_instance
                                in nested_instances)

                        # The parallel path we chose has more steps on the way
                        # to the leaf node, so we need to continue deeper into
                        # the tree.
                        if nested_slot_routes:
                            nested_provenances = tuple(
                                TemplateProvenanceNode(
                                    encloser_slot_key=nested_slot_name,
                                    encloser_slot_index=next(nested_index),
                                    instance_id=id(nested_instance),
                                    instance=nested_instance)
                                for nested_instance in nested_instances
                                if isinstance(
                                    nested_instance, nested_slot_type))
                            instance_history_stack.append(nested_provenances)
                            instance_backlog_stack.append(
                                deque(nested_provenances))
                            parallel_slot_backlog_stack.append(
                                list(nested_slot_routes))

                        # If there aren't any more nested slot routes, then
                        # we need to back out one level on the stack.
                        # Presumably this is also a terminus, but that's
                        # handled above.
                        else:
                            # Note that we already popped from the parallel
                            instance_backlog_stack[-1].popleft()

                # Oh the humanity, oh the combinatorics!
                invocations.extend(itertools.product(
                    provenances,
                    itertools.chain.from_iterable(
                        abstract_calls.values()
                    )))

        root_provenance = TemplateProvenance((root_provenance_node,))
        root_template_class = type(root_template_instance)
        invocations.extend(
            (root_provenance, abstract_call)
            for abstract_call
            in itertools.chain.from_iterable(
                template_preload[root_template_class].function_calls.values()
            ))
        return invocations

    def resolve_forward_ref(
            self,
            lookup_key: ForwardRefLookupKey,
            resolved_template_cls: TemplateClass
            ) -> None:
        """Notifies a dependent class (one that declared a slot as a
        forward reference) that the reference is now available, thereby
        causing it to resolve the forward ref and remove it from its
        pending trees.
        """
        resolved_template_xable = cast(
            type[TemplateIntersectable], resolved_template_cls)
        resolved_signature = resolved_template_xable._templatey_signature
        enclosing_template_cls = self.template_cls_ref()

        if enclosing_template_cls is None:
            raise RuntimeError(
                'Template class was garbage collected before template '
                + 'signature, and then signature asked to resolve forward '
                + 'ref?!')

        _resolve_pending_slot_tree(
            pending_tree=self._pending_ref_lookup.pop(lookup_key),
            slot_name=None,
            enclosing_cls=enclosing_template_cls,
            enclosing_slot_tree_lookup=self._slot_tree_lookup,
            enclosing_pending_ref_lookup=self._pending_ref_lookup,
            resolved_cls=resolved_template_cls,
            resolved_slot_tree_lookup=resolved_signature._slot_tree_lookup,
            resolved_pending_ref_lookup=resolved_signature._pending_ref_lookup)

        self.refresh_included_template_classes_snapshot()
        self.refresh_pending_forward_ref_registration()


def _extract_template_class_locals() -> dict[str, Any] | None:
    """When templates are created from inside a closure (ex, during
    testing, where this is extremely common), we need access to the
    locals from the closure to resolve type hints. This method relies
    upon ``inspect`` to extract them.

    Note that this can be very sensitive to where, exactly, you put it
    within the templatey code. Always put it as close as possible to
    the public API method, so that the first frame from another module
    coincides with the call to decorate a template class.
    """
    upmodule_frame = _get_first_frame_from_other_module()
    if upmodule_frame is not None:
        return upmodule_frame.f_locals


def _extract_frame_scope_id() -> int | None:
    """When templates are created from inside a closure (ex, during
    testing, where this is extremely common), and forward references
    are used, we need a way to differentiate between identically-named
    templates within different functions of the same module (or the
    toplevel of the module).

    We do this via a dedicated decorator/context manager,
    ``anchor_closure_scope``, which creates a random value and assigns
    it to the corresponding context var, and then is retrieved by this
    function for use.
    """
    return _CURRENT_SCOPE_ID.get()


def _get_first_frame_from_other_module() -> FrameType | None:
    """Both of our closure workarounds require walking up the stack
    until we reach the first frame coming from ^^outside^^ the ~~house~~
    current module. This performs that lookup.

    **Note that this is pretty fragile.** Or, put a different way: it
    does exactly what the function name suggest it does: it finds the
    FIRST frame from another module. That doesn't mean we won't return
    to this module; it doesn't mean it's from the actual client library,
    etc. It just means it's the first frame that isn't from this
    module.
    """
    upstack_frame = inspect.currentframe()
    if upstack_frame is None:
        return None
    else:
        this_module = upstack_module = inspect.getmodule(
            _extract_template_class_locals)
        while upstack_module is this_module:
            if upstack_frame is None:
                return None

            upstack_frame = upstack_frame.f_back
            upstack_module = inspect.getmodule(upstack_frame)

    return upstack_frame


def _classify_interface_field_flavor(
        parent_class_type_hints: dict[str, Any],
        template_field: Field
        ) -> tuple[InterfaceAnnotationFlavor, type] | None:
    """For a dataclass field, determines whether it was declared as a
    var, slot, or content.

    If none of the above, returns None.
    """
    # Note that dataclasses don't include the actual type (just a string)
    # when in __future__ mode, so we need to get them from the parent class
    # by calling get_type_hints() on it
    resolved_field_type = parent_class_type_hints[template_field.name]
    anno_origin = typing.get_origin(resolved_field_type)
    if anno_origin is Var:
        nested_type, = typing.get_args(resolved_field_type)
        return InterfaceAnnotationFlavor.VARIABLE, nested_type
    elif anno_origin is Slot:
        nested_type, = typing.get_args(resolved_field_type)
        return InterfaceAnnotationFlavor.SLOT, nested_type
    elif anno_origin is Content:
        nested_type, = typing.get_args(resolved_field_type)
        return InterfaceAnnotationFlavor.CONTENT, nested_type
    else:
        return None


@dataclass(frozen=True, slots=True)
class _PendingSlotTreeContainer:
    """Remember that the point here is to eventually build a lookup from
    a ``{type[template]: _SlotTreeNode}``. And we're dealing with
    forward references to the ``type[template]``, meaning we don't have
    a key to use for the slot tree lookup. End of story.

    So what we're doing here instead, is constructing the slot tree as
    best as we can, and keeping track of what nodes need to be populated
    by the forward reference, once it is resolved.

    When the forward ref is resolved, we can simple copy the tree into
    all of the insertion nodes, and then store the pending slot tree
    in the slot tree lookup using the resolved template class.
    """
    pending_slot_type: ForwardRefLookupKey
    pending_root_node: _PendingSlotTreeNode


class ForwardReferenceProxyClass(Protocol):
    REFERENCE_TARGET: ClassVar[ForwardRefLookupKey]


def is_forward_reference_proxy(
        obj: object
        ) -> TypeIs[type[ForwardReferenceProxyClass]]:
    return isinstance(obj, type) and hasattr(obj, 'REFERENCE_TARGET')


def _copy_slot_tree[T: _SlotTreeNode](
        src_tree: T,
        into_tree: T | None = None
        ) -> T:
    """This creates a copy of an existing slot tree. We use it when
    merging nested slot trees into enclosers; otherwise, we end up with
    a huge mess of "it's not clear what object holds which slot tree"
    that is very difficult to reason about. This is slightly more memory
    intensive, but... again, this is much, much easier to reason about.

    Take special note that this preserves reference cycles, which is a
    bit of a tricky thing.

    Note: if ``into_tree`` is provided, this copies inplace and returns
    the ``into_tree``. Otherwise, a new tree is created and returned.
    In both cases, we also return a lookup from
    ``{old_node.id_: copied_node}``.
    """
    copied_tree: T
    if into_tree is None:
        copied_tree = src_tree.empty_clone()
    else:
        copied_tree = into_tree
        into_tree.merge_fields_only(src_tree)

    # This converts ``old_node.id_`` to the new node instance; it's how we
    # implement copying reference cycles
    transmogrified_nodes: dict[int, T] = {src_tree.id_: copied_tree}
    copy_stack: list[_SlotTreeTraversalFrame[T, T]] = [_SlotTreeTraversalFrame(
        next_subtree_index=0,
        existing_subtree=copied_tree,
        insertion_subtree=src_tree)]

    while copy_stack:
        current_stack_frame = copy_stack[-1]
        if current_stack_frame.exhausted:
            copy_stack.pop()
            continue

        next_slot_route = current_stack_frame.insertion_subtree[
            current_stack_frame.next_subtree_index]
        next_slot_name, next_slot_type, next_subtree = next_slot_route
        # Do this ASAP so that we don't accidentally forget it somehow
        current_stack_frame.next_subtree_index += 1

        next_subtree_id = next_subtree.id_
        already_copied_node = transmogrified_nodes.get(next_subtree_id)
        # This could be either the first time we hit a recursive subtree,
        # or a non-recursive subtree.
        if already_copied_node is None:
            new_subtree = next_subtree.empty_clone()

            if next_subtree.requires_transmogrification:
                transmogrified_nodes[next_subtree_id] = new_subtree

            current_stack_frame.existing_subtree.append(
                _SlotTreeRoute.new(
                    next_slot_name,
                    next_slot_type,
                    new_subtree,))
            copy_stack.append(_SlotTreeTraversalFrame(
                next_subtree_index=0,
                existing_subtree=new_subtree,
                insertion_subtree=next_subtree))

        # We've hit a recursive subtree -- one that we've already copied --
        # which means we don't need to copy it again; instead, we just need to
        # transmogrify the reference so that the nested route refers back to
        # the original copied node.
        else:
            current_stack_frame.existing_subtree.append(
                _SlotTreeRoute.new(
                    next_slot_name,
                    next_slot_type,
                    already_copied_node,))

    return copied_tree


def _merge_into_slot_tree[T: _SlotTreeNode](
        existing_slot_type: TemplateClass,
        existing_tree: T,
        to_merge: T
        ) -> None:
    """This traverses the existing tree, merging in the slot_name and
    its subtrees into the correct locations in the existing slot tree,
    recursively.

    This is needed because unions can have slots with overlapping slot
    names, but we don't want to redo a ton of tree recursion.

    In theory, this might result in some edge case scenarios where two
    effectively identical templates within a union, one of which calls
    an env function and one of which doesn't, might actually result in
    some unnecessary calls to the env function during prepopulation.
    I'm honestly not sure either way; we'd need more testing to verify
    either way. The solution in that case would probably be some kind
    of instance check to verify just before running the function that
    it was actually the expected instance type / has the expected
    function calls.

    We return a lookup of ``{source_node.id_: dest_node}``, which can
    be used for recording and/or resolving pending forward refs.
    """
    transmogrified_nodes: dict[int, T] = {to_merge.id_: existing_tree}

    # Counterintuitive: since we're MERGING trees, the existing_subtree is
    # actually the DESTINATION, and the insertion_subtree the source!
    merge_stack: list[_SlotTreeTraversalFrame[T, T]] = [
        _SlotTreeTraversalFrame(
            next_subtree_index=0,
            existing_subtree=existing_tree,
            insertion_subtree=to_merge)]
    # Yes, in theory, this one specific operation of merging trees would be
    # faster if the trees were dicts instead of iterative structures. But
    # we're not optimizing for tree merging; we're optimizing for rendering!
    # And in that case, we're better off with a simple iterative structure.
    while merge_stack:
        current_stack_frame = merge_stack[-1]
        if current_stack_frame.exhausted:
            merge_stack.pop()
            continue

        existing_subtree = current_stack_frame.existing_subtree
        next_slot_route = current_stack_frame.insertion_subtree[
            current_stack_frame.next_subtree_index]
        next_slot_name, next_slot_type, next_subtree = next_slot_route
        # Do this ASAP so that we don't accidentally forget it somehow (also
        # because we want to use a continue statement in a second)
        current_stack_frame.next_subtree_index += 1

        next_subtree_id = next_subtree.id_
        already_merged_node = transmogrified_nodes.get(next_subtree_id)

        # For merging, we're going to handle the recursive subtree case first,
        # because it makes the rest of the logic cleaner
        if already_merged_node is not None:
            current_stack_frame.existing_subtree.append(
                _SlotTreeRoute.new(
                    next_slot_name, next_slot_type, already_merged_node))
            continue

        # This accomplishes two things: first, it culls an extra cycle from the
        # to_merge tree that would ultimately have the same effect. Secondly,
        # it ensures correct recursion when we're merging in the pending tree
        # from a class that used us as a forward reference, since the other
        # class won't yet be resolved.
        if next_slot_type is existing_slot_type:
            next_existing_subtree = existing_tree
            next_existing_subtree.is_recursive = True

            # Remember that next_slot_type is the existing_slot_type that we
            # started this whole thing off with. Therefore, this should be
            # impossible: True would imply that we'd already resolved this
            # recursive branch pointing at itself and are somehow revisiting it
            # afterwards
            if existing_subtree.has_route_for(next_slot_name, next_slot_type):
                raise RecursionError(
                    'Non-culled infinite recursion while merging templatey '
                    + 'slot trees!', next_slot_name, next_slot_type)

            else:
                next_existing_route = _SlotTreeRoute.new(
                    next_slot_name,
                    next_slot_type,
                    next_existing_subtree)
                existing_subtree.append(next_existing_route)

            # Note that we don't need to update transmogrification for two
            # reasons: first, because we added the root node at the very
            # beginning, and second, because we -- by definition -- cannot
            # have any deeper references to this part of the destination tree,
            # because  we're culling the depthwise-rest of the source tree
            continue

        # The existing subtree -- the one we're merging INTO -- has a route for
        # this slot name and type already, so we need to merge them together
        # instead of simply copy/transmogrify/cull
        elif existing_subtree.has_route_for(next_slot_name, next_slot_type):
            next_existing_route = existing_subtree.get_route_for(
                    next_slot_name, next_slot_type)
            __, __, next_existing_subtree = next_existing_route
            next_existing_subtree.merge_fields_only(next_subtree)

        # The existing subtree doesn't have any existing routes for this, so
        # we don't need to worry about merging things together -- but we still
        # need to worry about transmogrification and culling.
        # Also note that there might be identically-named slots for different
        # slot types in the case of a union, but that will be handled on a
        # different iteration of the merge stack while loop.
        else:
            next_existing_subtree = next_subtree.empty_clone()
            next_existing_route = _SlotTreeRoute.new(
                next_slot_name,
                next_slot_type,
                next_existing_subtree)
            existing_subtree.append(next_existing_route)

        if next_existing_subtree.requires_transmogrification:
            transmogrified_nodes[next_subtree_id] = next_existing_subtree

        if next_subtree:
            merge_stack.append(_SlotTreeTraversalFrame(
                next_subtree_index=0,
                existing_subtree=next_existing_subtree,
                insertion_subtree=next_subtree))


def _resolve_pending_slot_tree(  # noqa: PLR0913
        pending_tree: _PendingSlotTreeContainer,
        slot_name: str | None,
        enclosing_cls: TemplateClass,
        enclosing_slot_tree_lookup: dict[TemplateClass, _ConcreteSlotTreeNode],
        enclosing_pending_ref_lookup:
            dict[ForwardRefLookupKey, _PendingSlotTreeContainer],
        resolved_cls: TemplateClass,
        resolved_slot_tree_lookup: dict[TemplateClass, _ConcreteSlotTreeNode],
        resolved_pending_ref_lookup:
            dict[ForwardRefLookupKey, _PendingSlotTreeContainer],
        ) -> None:
    """Once a forward ref has been resolved, it needs to have both the
    concrete and pending trees from the now-resolved class merged into
    the respective trees of the enclosing class.

    There are several things to be mindful of here:
    ++  Pending trees are really factories for concrete slot trees. Each
        nested template class in the resolved class will be inserted
        at the pending tree insertion point. In other words, if there
        are N insertion points, and the resolved class has M slots, then
        N * M concretified slot trees will need to be merged into the
        enclosing class.
    ++  The slot tree of the resolved class (or its pending tree) may
        include references to the enclosing class. These subtrees need
        to then be short-circuited, and replaced by references to the
        slot tree root for the enclosing class.
    ++  The resolved class may not be completely finalized yet, since
        we are also called during initial tree creation for the resolved
        class (though only when recursive loops are encountered).
        Therefore, we need to explicitly pass the concrete and pending
        slot tree lookups for each of the enclosing and resolved
        classes, in addition to the class itself
    ++  The resolved and enclosing classes might be the same class (as
        in the above case, where this is called during initial tree
        creation for a resolved class in a recursive loop).
    """
    # We need to make sure to include the actual resolved class itself in
    # the slot tree! But instead of special-casing it, we can just let it be
    # handled by the same logic for the nested ones.
    resolved_slot_item = (resolved_cls, _SlotTreeNode(is_terminus=True))
    all_nested_slot_items = itertools.chain(
        resolved_slot_tree_lookup.items(),
        [resolved_slot_item])

    pending_tree_root_node = pending_tree.pending_root_node
    for nested_template_cls,  nested_root_node in all_nested_slot_items:
        # Note: this means that we're resolving against the root tree. That
        # means that the pending tree is relative to it, and not to a nested
        # slot. This implies that we're resolving a forward ref on an
        # already-existing template class.
        if slot_name is None:
            tree_after_insertions = _apply_insertions(
                resolved_cls,
                pending_tree_root_node,
                nested_root_node)

        # However, in this case, we're resolving against a specific slot on
        # a root key. This implies that we're resolving a forward ref on a
        # not-yet-exited-the-decorator template class.
        else:
            # Note that the nesting is crucial; that's what puts the tree under
            # the correct slot for the enclosing class, instead of retaining
            # the tree depth and root slot from the resolved class.
            tree_after_insertions = _SlotTreeNode([_SlotTreeRoute.new(
                slot_name,
                resolved_cls,
                _apply_insertions(
                    resolved_cls,
                    pending_tree_root_node,
                    nested_root_node))])

        dest_tree = enclosing_slot_tree_lookup.get(nested_template_cls)
        # Note that we still need to call merge_into to perform any needed
        # culling, so we can't simply assign the dest_tree as the value in the
        # enclosing lookup
        if dest_tree is None:
            dest_tree = enclosing_slot_tree_lookup[nested_template_cls] = (
                _SlotTreeNode())

        _merge_into_slot_tree(
            enclosing_cls,
            dest_tree,
            tree_after_insertions)

    for (
        forward_ref_key,
        pending_tree_container
    ) in resolved_pending_ref_lookup.items():
        # Note: this means that we're resolving against the root tree. That
        # means that the pending tree is relative to it, and not to a nested
        # slot. This implies that we're resolving a forward ref on an
        # already-existing template class.
        if slot_name is None:
            tree_after_insertions = _apply_insertions(
                resolved_cls,
                pending_tree_root_node,
                pending_tree_container.pending_root_node)

        # However, in this case, we're resolving against a specific slot on
        # a root key. This implies that we're resolving a forward ref on a
        # not-yet-exited-the-decorator template class.
        else:
            # Again, nesting here is crucial, otherwise the tree would be
            # missing its slot name for the enclosing_cls
            tree_after_insertions = _PendingSlotTreeNode([_SlotTreeRoute.new(
                slot_name,
                resolved_cls,
                _apply_insertions(
                    resolved_cls,
                    pending_tree_root_node,
                    pending_tree_container.pending_root_node))])

        dest_pending = enclosing_pending_ref_lookup.get(forward_ref_key)
        # Note that we still need to call merge_into to perform any needed
        # culling, so we can't simply assign the dest_tree as the value in the
        # enclosing lookup
        if dest_pending is None:
            dest_tree =  _PendingSlotTreeNode()
            enclosing_pending_ref_lookup[forward_ref_key] = (
                _PendingSlotTreeContainer(forward_ref_key, dest_tree))
        else:
            dest_tree = dest_pending.pending_root_node

        _merge_into_slot_tree(
            enclosing_cls,
            dest_tree,
            tree_after_insertions)


def _apply_insertions[T: _ConcreteSlotTreeNode | _PendingSlotTreeNode](
        resolved_cls: TemplateClass,
        pending_tree: _PendingSlotTreeNode,
        nested_root_node: T,
        ) -> T:
    """For a particular pending tree and a single nested CONCRETE root
    node, this searches the pending tree for all insertion points and
    then inserts the concrete root node at that point.

    **Note that this does not cull any superfluous recursive
    refererence cycles.** You still need to merge the resulting tree
    into the actual slot tree lookup, and then that does the culling.
    """
    resulting_tree: T = type(nested_root_node)()
    resulting_tree.merge_fields_only(pending_tree)
    # This converts ``old_node.id_`` to the new node instance; it's how we
    # implement copying reference cycles
    transmogrified_nodes: dict[int, _SlotTreeNode] = {
        pending_tree.id_: resulting_tree}

    stack: \
        list[_SlotTreeTraversalFrame[_SlotTreeNode, _PendingSlotTreeNode]] = [
        _SlotTreeTraversalFrame(
            next_subtree_index=0,
            existing_subtree=resulting_tree,
            insertion_subtree=pending_tree)]

    while stack:
        current_stack_frame = stack[-1]
        if current_stack_frame.exhausted:
            stack.pop()
            continue

        src_subtree = current_stack_frame.insertion_subtree
        target_subtree = current_stack_frame.existing_subtree
        next_slot_route = src_subtree[current_stack_frame.next_subtree_index]
        next_slot_name, next_slot_type, next_subtree = next_slot_route
        # Do this ASAP so that we don't accidentally forget it somehow
        current_stack_frame.next_subtree_index += 1

        # Note that this will still get merged into the actual full slot tree
        # for the enclosing template, which will cull any extra links in
        # recursive reference cycles, so we don't need to worry about that
        # here.
        next_subtree_id = next_subtree.id_
        already_applied_node = transmogrified_nodes.get(next_subtree_id)

        # This could be either the first time we hit a recursive subtree,
        # or a non-recursive subtree.
        if already_applied_node is None:
            new_subtree = _SlotTreeNode()
            new_subtree.merge_fields_only(next_subtree)

            if next_subtree.requires_transmogrification:
                transmogrified_nodes[next_subtree_id] = new_subtree

            target_subtree.append(
                _SlotTreeRoute.new(
                    next_slot_name,
                    next_slot_type,
                    new_subtree))
            stack.append(_SlotTreeTraversalFrame(
                next_subtree_index=0,
                existing_subtree=new_subtree,
                insertion_subtree=next_subtree))

        # We've hit a recursive subtree -- one that we've already copied --
        # which means we don't need to copy it again; instead, we just need to
        # transmogrify the reference so that the nested route refers back to
        # the original copied node.
        else:
            # Note: is_recursive was already set!
            target_subtree.append(
                _SlotTreeRoute.new(
                    next_slot_name,
                    next_slot_type,
                    already_applied_node))

        # Note that this is IN ADDITION to copying any nested routes! This is
        # purely for the insertions.
        for insertion_slot_name in src_subtree.insertion_slot_names:
            target_subtree.append(
                _SlotTreeRoute.new(
                    insertion_slot_name,
                    resolved_cls,
                    _copy_slot_tree(nested_root_node)))

    return resulting_tree


@dataclass(slots=True)
class _SlotTreeTraversalFrame[ET: _SlotTreeNode, IT: _SlotTreeNode]:
    next_subtree_index: int
    existing_subtree: ET
    insertion_subtree: IT

    @property
    def exhausted(self) -> bool:
        """Returns True if the to_merge_subtree has been exhausted, and
        there are no more subtrees to merge.
        """
        return self.next_subtree_index >= len(self.insertion_subtree)


# Note: mutablemapping because otherwise chainmap complains. Even though they
# aren't actually implemented, this is a quick way of getting typing to work
@dataclass(kw_only=True, slots=True)
class _ForwardRefGeneratingNamespaceLookup(MutableMapping[str, type]):
    template_module: str
    template_scope_id: int | None
    captured_refs: set[ForwardRefLookupKey] = field(default_factory=set)

    def __getitem__(self, key: str) -> type:
        forward_ref = ForwardRefLookupKey(
            module=self.template_module,
            name=key,
            scope_id=self.template_scope_id)

        class ForwardReferenceProxyClass:
            """When we return a forward reference, we want to retain all
            of the expected behavior with types -- unions via ``|``,
            etc -- and therefore, we want to return a proxy class
            instead of the forward reference itself.
            """
            REFERENCE_TARGET = forward_ref

        self.captured_refs.add(forward_ref)
        return ForwardReferenceProxyClass

    # Required for mutable mapping protocol, but not for the namespace lookup.
    def __iter__(self) -> Iterator[str]:
        raise TypeError(
            'Unsupported method call in templatey foward ref implementation.')

    # Required for mutable mapping protocol, but not for the namespace lookup.
    def __len__(self) -> int:
        raise TypeError(
            'Unsupported method call in templatey foward ref implementation.')

    # Required for mutable mapping protocol, but not for the namespace lookup.
    def __setitem__(self, key, value) -> None:
        raise TypeError(
            'Unsupported method call in templatey foward ref implementation.')

    # Required for mutable mapping protocol, but not for the namespace lookup.
    def __delitem__(self, key) -> None:
        raise TypeError(
            'Unsupported method call in templatey foward ref implementation.')


@dataclass_transform(field_specifiers=(param, field, Field))
def make_template_definition[T: type](
        cls: T,
        *,
        dataclass_kwargs: dict[str, bool],
        # Note: needs to be understandable by template loader
        template_resource_locator: object,
        template_config: TemplateConfig
        ) -> T:
    """Programmatically creates a template definition. Converts the
    requested class into a dataclass, passing along ``dataclass_kwargs``
    to the dataclass constructor. Then performs some templatey-specific
    bookkeeping. Returns the resulting dataclass.
    """
    cls = dataclass(**dataclass_kwargs)(cls)
    cls._templatey_config = template_config
    cls._templatey_resource_locator = template_resource_locator

    template_module = cls.__module__
    template_scope_id = _extract_frame_scope_id()
    template_forward_ref = ForwardRefLookupKey(
        module=template_module,
        name=cls.__name__,
        scope_id=template_scope_id)

    # We're prioritizing the typical case here, where the templates are defined
    # at the module toplevel, and therefore accessible within the module
    # globals. However, if the template is defined within a closure, we might
    # need to walk up the stack until we find a caller that isn't within this
    # file, and then grab its locals.
    try:
        template_type_hints = typing.get_type_hints(cls)
    except NameError as exc:
        logger.info(dedent('''\
            Failed to resolve template type hints on first pass. This could be
            indicative of a bug, or it might occur in normal situations if:
            ++  you're defining the template within a closure. Here, we'll
                attempt to infer the locals via inspect.currentframe, but not
                all platforms support that, which can lead to failures
            ++  the type hint is a forward reference.

            In both cases, we'll wrap the request into a
            ``ForwardRefLookupKey``, which will then hopefully be
            resolved as soon as the forward reference is declared.
            If it's never resolved, however, we will raise whenever
            ``render`` is called.
            '''),
            exc_info=exc)

        # There's a method to the madness here.
        # globalns needs to be strictly a dict, because it gets delegated into
        # ``eval``, which requires one. Which means we can only use the localns
        # to intercept missing forward references. But that, then, means that
        # we need to recover the existing check for the actual globals, since
        # otherwise **all** global names would be overwritten by the forward
        # reference.
        forwardref_lookup = _ForwardRefGeneratingNamespaceLookup(
            template_module=template_module,
            template_scope_id=template_scope_id)
        # This is the same as the current implementation of get_type_hints
        # in cpython for classes:
        # https://github.com/python/cpython/blob/0045100ccbc3919e8990fa59bc413fe38d21b075/Lib/typing.py#L2325
        template_globals = getattr(
            sys.modules.get(template_module, None), '__dict__', {})

        maybe_locals = _extract_template_class_locals()
        if maybe_locals is None:
            prioritized_lookups = (
                template_globals,
                # Fun fact: these aren't included in the other globals!
                __builtins__,
                forwardref_lookup)

        else:
            prioritized_lookups = (
                maybe_locals,
                template_globals,
                # Fun fact: these aren't included in the other globals!
                __builtins__,
                forwardref_lookup)

        # Because of our forward lookup, this will always succeed
        template_type_hints = typing.get_type_hints(
            cls, localns=ChainMap(*prioritized_lookups))

    slots = {}
    vars_ = {}
    content = {}
    prerenderers = {}
    for template_field in fields(cls):
        field_classification = _classify_interface_field_flavor(
            template_type_hints, template_field)

        # Note: it's not entirely clear to me that this restriction makes
        # sense; I could potentially see MAYBE there being some kind of
        # environment function that could access other attributes from the
        # dataclass? But also, maybe those should be vars? Again, unclear.
        if field_classification is None:
            raise TypeError(
                'Template parameter definitions may only contain variables, '
                + 'slots, and content!')

        else:
            field_flavor, wrapped_type = field_classification

            # A little awkward to effectively just repeat the comparison we did
            # when classifying, but that makes testing easier and control flow
            # clearer
            if field_flavor is InterfaceAnnotationFlavor.VARIABLE:
                dest_lookup = vars_
            elif field_flavor is InterfaceAnnotationFlavor.SLOT:
                dest_lookup = slots
            else:
                dest_lookup = content

            dest_lookup[template_field.name] = wrapped_type
            prerenderers[template_field.name] = template_field.metadata.get(
                'templatey.prerenderer')

    cls._templatey_signature = TemplateSignature.new(
        template_cls=cls,
        slots=slots,
        vars_=vars_,
        content=content,
        forward_ref_lookup_key=template_forward_ref)
    converter_cls = namedtuple('TemplateyConverters', tuple(prerenderers))
    cls._templatey_prerenderers = converter_cls(**prerenderers)

    # Note: this needs to be the absolute last thing, because we need to fully
    # satisfy the intersectable interface before we can call it.
    _resolve_forward_references(cls)
    return cls


def _resolve_forward_references(pending_template_cls: TemplateClass):
    """The very last thing to do before we return the class after
    template decoration is to resolve all forward references inside the
    class. To do that, we first need to construct the corresponding
    ForwardRefLookupKey and check for it in the pending forward refs
    lookup.

    If we find one, we then need to update the values there, while
    checking for and correctly handling recursion.
    """
    lookup_key = ForwardRefLookupKey(
        module=pending_template_cls.__module__,
        name=pending_template_cls.__name__,
        scope_id=_extract_frame_scope_id())

    forward_ref_registry = _PENDING_FORWARD_REFS.get()
    dependent_template_classes = forward_ref_registry.get(lookup_key)
    if dependent_template_classes is not None:
        for dependent_template_cls in dependent_template_classes:
            dependent_xable = cast(
                TemplateIntersectable, dependent_template_cls)
            dependent_xable._templatey_signature.resolve_forward_ref(
                lookup_key, pending_template_cls)

        del forward_ref_registry[lookup_key]


@dataclass(frozen=True, slots=True)
class InjectedValue:
    """This is used by environment functions and complex content to
    indicate that a value is being injected into the template. Use it
    instead of a bare string to preserve an existing interpolation
    config, or to indicate whether verification and/or escaping should
    be applied to the value after conversion to a string.

    Note that, if both are defined, the variable escaper will be called
    first, before the content verifier.
    """
    value: object

    config: InterpolationConfig = field(default_factory=InterpolationConfig)
    use_content_verifier: bool = False
    use_variable_escaper: bool = True

    def __post_init__(self):
        if self.config.prefix is not None or self.config.suffix is not None:
            raise ValueError(
                'Injected values cannot have prefixes nor suffixes. If you '
                + 'need similar behavior, simply add the affix(es) to the '
                + 'iterable returned by the complex content or env function.')


class _ComplexContentBase(Protocol):

    def flatten(
            self,
            dependencies: Annotated[
                Mapping[str, object],
                ClcNote(
                    '''The values of the variables declared as dependencies
                    in the constructor are passed to the call to ``flatten``
                    during rendering.
                    ''')],
            config: Annotated[
                InterpolationConfig,
                ClcNote(
                    '''The interpolation configuration of the content
                    interpolation that the complex content is a member
                    of. Note that neither prefix nor suffix can be passed
                    on to an ``InjectedValue``; they must be manually included
                    in the return value if desired.
                    ''')],
            prerenderers: Annotated[
                Mapping[str, InterpolationPrerenderer | None],
                ClcNote(
                    '''If a prerenderer is defined on a dependency variable,
                    it will be included here; otherwise, the value will be
                    set to ``None``.
                    ''')],
            ) -> Iterable[object | InjectedValue]:
        """Implement this for any instance of complex content.

        First, do whatever content modification you need to, based on
        the dependency variables declared in the constructor. Then,
        if needed, merge in the variable itself using an
        ``InjectedValue``, configuring it as appropriate.

        **Note that the parent interpolation config will be ignored by
        all strings returned by flattening individually.** So if, for
        example, you included a prefix in the content interpolation
        within the template itself, and then passed a ``ComplexContent``
        instance to the template instance, the prefix would be ignored
        completely (unless you do something with it in ``flatten``).

        **Also note that you are responsible for calling the dependency
        variable's ``InterpolationPrerenderer``. directly,** within your
        implementation of ``flatten``. This affords you the option to
        skip it if desired.

        > Example: noun quantity
        __embed__: 'code/python'
            class NaivePluralContent(ComplexContent):

                def flatten(
                        self,
                        dependencies: Mapping[str, object],
                        config: InterpolationConfig,
                        prerenderers:
                            Mapping[str, InterpolationPrerenderer | None],
                        ) -> Iterable[str | InjectedValue]:
                    \"""Pluralizes the name of the provided dependency.
                    For example, ``{'widget': 1}`` will be rendered as
                    "1 widget", but ``{'widget': 2}`` will be rendered as
                    "2 widgets".
                    \"""

                    # Assume only 1 dependency
                    name, value = next(iter(dependencies.items()))

                    if 0 <= value <= 1:
                        return (
                            InjectedValue(
                                value,
                                # This assumes no prefix/suffix
                                config=config,
                                use_content_verifier=False,
                                use_variable_escaper=True),
                            ' ',
                            name)

                    else:
                        return (
                            InjectedValue(
                                value,
                                # This assumes no prefix/suffix
                                config=config,
                                use_content_verifier=False,
                                use_variable_escaper=True),
                            ' ',
                            name,
                            's')
        """
        ...


@dataclass(slots=True, kw_only=True)
class ComplexContent(_ComplexContentBase):
    """Sometimes content isn't as simple as a ``string``. For example,
    content might include variable interpolations. Or you might need
    to modify the content slightly based on the variables -- for
    example, to get subject/verb alignment based on a number, gender
    alignment based on a pronoun, or whatever. ComplexContent gives
    you an escape hatch to do this: simply pass a ComplexContent
    instance as a value instead of a string.
    """

    dependencies: Annotated[
        Collection[str],
        ClcNote(
            '''Complex content dependencies are the **variable** names
            that a piece of complex content depends on. These will be
            passed to the implemented ``flatten`` function during
            rendering.
            ''')]
