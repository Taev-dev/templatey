from __future__ import annotations

import functools
import inspect
import itertools
import logging
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
from dataclasses import _MISSING_TYPE
from dataclasses import Field
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from itertools import tee as tee_iterable
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
from typing import cast
from typing import dataclass_transform
from typing import overload

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
logger = logging.getLogger(__name__)


type TemplateClass = type[TemplateParamsInstance]
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

# Technically 3.10 or better, but we require that anyways
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
    have multiple provenance nodes. And each different slot in a parent
    will have a separate provenance node, potentially with different
    namespace overrides.

    This is used both during function execution (to calculate the
    concrete value of any parameters), and also while flattening the
    render tree.

    Also note that the root template, **along with any templates
    injected into the render by environment functions,** will have an
    empty list of provenance nodes.

    Note that, since overrides from the parent can come exclusively from
    the template body -- and are therefore shared across all children of
    the same slot -- they don't get stored within the provenance, since
    we'd require access to the template bodies, which we don't yet have.
    """
    parent_slot_key: str
    parent_slot_index: int
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
        parent_param_name = name
        parent_slot_key = current_provenance_node.parent_slot_key
        for parent in reversed(self[0:-1]):
            template_class = type(parent.instance)
            parent_template = template_preload[template_class]
            parent_overrides = parent_template.slots[parent_slot_key].params

            if parent_param_name in parent_overrides:
                value = parent_overrides[parent_param_name]

                if isinstance(value, NestedContentReference):
                    parent_slot_key = parent.parent_slot_key
                    parent_param_name = value.name
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
        parent_param_name = name
        parent_slot_key = current_provenance_node.parent_slot_key
        for parent in reversed(self[0:-1]):
            template_class = type(parent.instance)
            parent_template = template_preload[template_class]
            parent_overrides = parent_template.slots[parent_slot_key].params

            if parent_param_name in parent_overrides:
                value = parent_overrides[parent_param_name]

                if isinstance(value, NestedVariableReference):
                    parent_slot_key = parent.parent_slot_key
                    parent_param_name = value.name
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


# Note that the string annotation here is required because of the forward ref,
# despite the __future__ import
class _SlotTreeNode(list['_SlotTreeRoute']):
    """The purpose of the slot tree is to precalculate what sequences of
    getattr() calls we need to traverse to arrive at every instance of a
    particular slot type for a given template, including all nested
    templates.

    **These are optimized for rendering, not for template declaration.**

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
    # We use this to make the logic cleaner when merging trees, but we want
    # the faster performance of the tuple when actually traversing the tree
    _routes_by_slot_name: dict[str, _SlotTreeRoute]
    # We use this to limit the number of entries we need in the transmogrifier
    # lookup during tree merging/copying
    is_recursive: bool
    # We use this to differentiate between dissimilar unions, one which
    # continues on to a subtree, and one which ends here with the target
    # instance
    is_terminus: bool

    def __new__(
            cls,
            routes: Iterable[_SlotTreeRoute] | None = None,
            *,
            is_recursive: bool = False,
            is_terminus: bool = False,
            ) -> _SlotTreeNode:
        if routes is None:
            self = super().__new__(cls)
        else:
            self = super().__new__(cls, routes)

        self.is_recursive = is_recursive
        self.is_terminus = is_terminus
        self._routes_by_slot_name = {route[0]: route for route in self}
        return self

    def append(self, route: _SlotTreeRoute):
        slot_name = route[0]
        if slot_name in self._routes_by_slot_name:
            raise ValueError(
                'Templatey internal error: attempt to append duplicate slot '
                + 'name! Please search for / report issue to github along '
                + 'with a traceback.')

        super().append(route)
        self._routes_by_slot_name[slot_name] = route

    def extend(self, routes: Iterable[_SlotTreeRoute]):
        routes, routes_copy_1, routes_copy_2 = tee_iterable(routes, 3)

        if any(
            route_copy[0] in self._routes_by_slot_name
            for route_copy in routes_copy_1
        ):
            raise ValueError(
                'Templatey internal error: attempt to append duplicate slot '
                + 'name! Please search for / report issue to github along '
                + 'with a traceback.')

        super().extend(routes)
        self._routes_by_slot_name.update({
            route_copy[0]: route_copy for route_copy in routes_copy_2})

    def has_route_for(self, slot_name: str) -> bool:
        return slot_name in self._routes_by_slot_name

    def get_route_for(self, slot_name: str) -> _SlotTreeRoute:
        return self._routes_by_slot_name[slot_name]


class _SlotTreeRoute(tuple[str, _SlotTreeNode]):
    """An individual route on the slot tree is defined by the attribute
    name for the slot, and the subtree from the slot class.

    TODO: add an explicit terminus to support partially-dissimilar
    unions.
    """
    # We use this to detect recursion loops when merging trees.
    # Note: each route is a different slot, and therefore usually also a
    # different slot class, but it might be a union, which we convert into
    # a set.
    slot_types: set[type[TemplateParamsInstance] | ForwardReferenceParam]

    def __new__(
            cls,
            slot_name: str,
            subtree: _SlotTreeNode,
            *,
            slot_types:
                set[type[TemplateParamsInstance] | ForwardReferenceParam]
            ) -> _SlotTreeRoute:
        self = super().__new__(cls, (slot_name, subtree))
        self.slot_types = slot_types

        return self

    @classmethod
    def from_slot_type(
            cls,
            slot_name: str,
            subtree: _SlotTreeNode,
            *,
            # Note: Unions automatically get converted into a set.
            slot_type:
                type[TemplateParamsInstance]
                | ForwardReferenceParam
                | UnionType,
            ) -> _SlotTreeRoute:

        if isinstance(slot_type, UnionType):
            slot_types = set(slot_type.__args__)
        else:
            slot_types = {slot_type}

        return cls(slot_name, subtree, slot_types=slot_types)


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
    slot_names: frozenset[str]
    var_names: frozenset[str]
    content_names: frozenset[str]
    # Note that this gets updated when forward references are resolved, so
    # it's easiest to keep it a set instead of a frozenset
    included_template_classes: set[type[TemplateParamsInstance]]

    # Note that these contain all included types, not just the ones on the
    # outermost layer that are associated with the signature. In other words,
    # they include the flattened recursion of all included slots, all the way
    # down the tree
    _slot_tree_lookup: dict[type[TemplateParamsInstance], _SlotTreeNode]
    _pending_ref_lookup: dict[ForwardReferenceParam, _PendingSlotTree]

    @classmethod
    def new(
            cls,
            for_cls: type,
            slots: dict[str,
                type[TemplateParamsInstance]
                | UnionType
                | type[ForwardReferenceProxyClass]],
            vars_: dict[str, type | type[ForwardReferenceProxyClass]],
            content: dict[str, type | type[ForwardReferenceProxyClass]]
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
        tree_wip: dict[type[TemplateParamsInstance], _SlotTreeNode]
        tree_wip = defaultdict(_SlotTreeNode)
        pending_ref_lookup: dict[ForwardReferenceParam, _PendingSlotTree] = {}
        for slot_name, slot_annotation in slots.items():
            cls._extend_wip_slot_and_ref_trees(
                for_cls,
                slot_name,
                slot_annotation,
                slot_tree_lookup=tree_wip,
                pending_ref_lookup=pending_ref_lookup)

        tree_wip.default_factory = None

        return cls(
            slot_names=slot_names,
            var_names=var_names,
            content_names=content_names,
            _slot_tree_lookup=tree_wip,
            _pending_ref_lookup=pending_ref_lookup,
            included_template_classes=set(tree_wip))

    @staticmethod
    def _extend_wip_slot_and_ref_trees(
            # This is used as a recursion guard. If we have a simple recursion,
            # there (somewhat surprisingly) isn't a forward ref at all when
            # getting the type hints, but instead, the actual class. But we're
            # still in the middle of populating its xable attributes, so we
            # need to short circuit it.
            for_cls: type,
            slot_name: str,
            slot_annotation:
                type[TemplateParamsInstance]
                | UnionType
                | type[ForwardReferenceProxyClass],
            *,
            slot_tree_lookup: defaultdict[
                type[TemplateParamsInstance], _SlotTreeNode],
            pending_ref_lookup: dict[ForwardReferenceParam, _PendingSlotTree]
            ) -> None:
        """Builds the slot tree for a single slot on a template class.
        Returns a tuple of (fully resolved, pending forward reference)
        lookups.

        Keep in mind that child slots might themselves include pending
        trees, so we can't infer based on the annotation type whether
        or not the result will include them or not.
        """
        slot_types: Collection[
            type[TemplateParamsInstance]
            | type[ForwardReferenceProxyClass]]

        if isinstance(slot_annotation, UnionType):
            slot_types = slot_annotation.__args__
        else:
            slot_types = (slot_annotation,)

        for slot_type in slot_types:
            if is_forward_reference_proxy(slot_type):
                forward_ref = slot_type.REFERENCE_TARGET
                dest_insertion = _SlotTreeNode()
                dest_slot_route = _SlotTreeRoute.from_slot_type(
                    slot_name,
                    dest_insertion,
                    slot_type=forward_ref)

                existing_pending_tree = pending_ref_lookup.get(forward_ref)
                if existing_pending_tree is None:
                    pending_ref_lookup[forward_ref] = _PendingSlotTree(
                        pending_slot_type=forward_ref,
                        pending_root_node=_SlotTreeNode([dest_slot_route]),
                        insertion_nodes=[dest_insertion])

                else:
                    existing_pending_tree.pending_root_node.append(
                        dest_slot_route)
                    existing_pending_tree.insertion_nodes.append(
                        dest_insertion)

            # In the simple recursion case -- a template defines a slot of its
            # own class -- we can immediately create a reference loop without
            # needing a forward ref.
            elif slot_type is for_cls:
                recursive_slot_tree = slot_tree_lookup[slot_type]
                recursive_slot_tree.is_recursive = True
                recursive_slot_route = _SlotTreeRoute.from_slot_type(
                    slot_name,
                    recursive_slot_tree,
                    slot_type=slot_type)
                recursive_slot_tree.append(recursive_slot_route)

            else:
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
                        slot_tree_lookup[nested_slot_type],
                        slot_name,
                        nested_slot_type,
                        nested_slot_tree)

                # Okay, now all we have left to do is transform all of the
                # pending references on the child into pending references on
                # the enclosing template.
                for (
                    nested_ref_param, nested_pending_slot_tree
                ) in nested_pending_refs.items():
                    existing_pending_tree = pending_ref_lookup.get(
                        nested_ref_param)

                    if existing_pending_tree is None:
                        # Transmogrified nodes gives us a lookup from the old
                        # IDs to the new insertion nodes, allowing us to
                        # convert the references to the copy.
                        copied_tree, transmogrified_nodes = _copy_slot_tree(
                            nested_pending_slot_tree.pending_root_node)
                        transmogrified_insertion_nodes = [
                            transmogrified_nodes[id(precopy_insertion_node)]
                            for precopy_insertion_node
                            in nested_pending_slot_tree.insertion_nodes
                            if precopy_insertion_node.is_recursive]
                        pending_ref_lookup[nested_ref_param] = (
                            _PendingSlotTree(
                                pending_root_node=_SlotTreeNode(
                                    [_SlotTreeRoute.from_slot_type(
                                        slot_name,
                                        copied_tree,
                                        slot_type=nested_ref_param)]),
                                insertion_nodes=transmogrified_insertion_nodes,
                                pending_slot_type=nested_ref_param)
                        )

                    else:
                        transmogrified_nodes = _merge_into_slot_tree(
                            existing_pending_tree.pending_root_node,
                            root_slot_name=slot_name,
                            to_merge_slot_type=
                                nested_pending_slot_tree.pending_slot_type,
                            to_merge=nested_pending_slot_tree.pending_root_node
                        )
                        transmogrified_insertion_nodes = [
                            transmogrified_nodes[id(precopy_insertion_node)]
                            for precopy_insertion_node
                            in nested_pending_slot_tree.insertion_nodes]
                        existing_pending_tree.insertion_nodes.extend(
                            transmogrified_insertion_nodes)

                # Note that the empty node here denotes that it doesn't have
                # any children **for the current node.** That doesn't mean that
                # the child tree doesn't have any other slots of the same type
                # (hence using append), but we're mapping ALL of the nodes, and
                # NOT just the leaves.
                slot_tree_lookup[slot_type].append(
                    _SlotTreeRoute.from_slot_type(
                        slot_name,
                        _SlotTreeNode(is_terminus=True),
                        slot_type=slot_type))

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
        # ++  parent template classes can have multiple slots with the same
        #     child template class. We call these "parallel slots" in this
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
        child_slot_name: str
        child_slot_routes: list[_SlotTreeRoute]
        child_instances: Sequence[TemplateParamsInstance]

        # This is used in multiple loop iterations, plus at the end to add any
        # function calls for the root instance.
        root_provenance_node = TemplateProvenanceNode(
            parent_slot_key='',
            parent_slot_index=-1,
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
                        child_slot_name, child_slot_routes = (
                            parallel_slot_backlog_stack[-1][-1])
                        current_instance = (
                            instance_backlog_stack[-1][0].instance)
                        child_instances = getattr(
                            # Note: the default here is necessary because of
                            # type unions; we need to support the case where
                            # the two classes in the union have different
                            # slot names
                            current_instance, child_slot_name, ())
                        child_index = itertools.count()

                        # The parallel path we chose has more steps on the way
                        # to the leaf node, so we need to continue deeper into
                        # the tree.
                        if child_slot_routes:
                            child_provenances = tuple(
                                TemplateProvenanceNode(
                                    parent_slot_key=child_slot_name,
                                    parent_slot_index=next(child_index),
                                    instance_id=id(child_instance),
                                    instance=child_instance)
                                for child_instance in child_instances)
                            instance_history_stack.append(child_provenances)
                            instance_backlog_stack.append(
                                deque(child_provenances))
                            parallel_slot_backlog_stack.append(
                                list(child_slot_routes))

                        # The parallel path we chose is actually a leaf node,
                        # which means that each child instance is a provenance.
                        else:
                            partial_provenance = tuple(
                                instance_backlog_level[0]
                                for instance_backlog_level
                                in instance_backlog_stack)

                            # Note that using extend here is basically just a
                            # shorthand for repeatedly iterating on the
                            # outermost while loop after appending the
                            # children (like we did with child_slot_routes)
                            provenances.extend(TemplateProvenance(
                                (
                                    *partial_provenance,
                                    TemplateProvenanceNode(
                                        parent_slot_key=child_slot_name,
                                        parent_slot_index=next(child_index),
                                        instance_id=id(child_instance),
                                        instance=child_instance)))
                                for child_instance
                                in child_instances)

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


def _extract_frame_scope() -> FrameType | None:
    """When templates are created from inside a closure (ex, during
    testing, where this is extremely common), and forward references
    are used, we need a way to differentiate between identically-named
    templates within different functions of the same module (or the
    toplevel of the module). We do this by including the frame object
    on the ForwardReferenceParam whenever we're in a closure. This
    method relies upon ``inspect`` to extract it.

    Note that this can be very sensitive to where, exactly, you put it
    within the templatey code. Always put it as close as possible to
    the public API method, so that the first frame from another module
    coincides with the call to decorate a template class.
    """
    upmodule_frame = _get_first_frame_from_other_module()
    if upmodule_frame is not None:
        frame_info = inspect.getframeinfo(upmodule_frame)

        # Handily enough, cpython uses this to indicate that we're not inside
        # a function currently, which is exactly what we need.
        if frame_info.function != '<module>':
            return upmodule_frame


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
class _PendingSlotTree:
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
    pending_slot_type: ForwardReferenceParam
    pending_root_node: _SlotTreeNode
    insertion_nodes: list[_SlotTreeNode]


@dataclass(frozen=True, slots=True)
class ForwardReferenceParam:
    """We use this to find all possible ForwardReferenceParam instances
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
    scope: FrameType | None


class ForwardReferenceProxyClass(Protocol):
    REFERENCE_TARGET: ClassVar[ForwardReferenceParam]


def is_forward_reference_proxy(
        obj: object
        ) -> TypeIs[type[ForwardReferenceProxyClass]]:
    return isinstance(obj, type) and hasattr(obj, 'REFERENCE_TARGET')


def _copy_slot_tree(
        src_tree: _SlotTreeNode,
        into_tree: _SlotTreeNode | None = None
        ) -> tuple[_SlotTreeNode, dict[int, _SlotTreeNode]]:
    """This creates a copy of an existing slot tree. We use it when
    merging nested slot trees into parents; otherwise, we end up with
    a huge mess of "it's not clear what object holds which slot tree"
    that is very difficult to reason about. This is slightly more memory
    intensive, but... again, this is much, much easier to reason about.

    Take special note that this preserves reference cycles, which is a
    bit of a tricky thing.

    Note: if ``into_tree`` is provided, this copies inplace and returns
    the ``into_tree``. Otherwise, a new tree is created and returned.
    In both cases, we also return a lookup from
    ``{id(old_node): copied_node}``.
    """
    copied_tree: _SlotTreeNode
    if into_tree is None:
        copied_tree = _SlotTreeNode(
            is_recursive=src_tree.is_recursive,
            is_terminus=src_tree.is_terminus)
    else:
        copied_tree = into_tree
        copied_tree.is_recursive |= src_tree.is_recursive
        copied_tree.is_terminus |= src_tree.is_terminus

    # This converts ``id(old_node)`` to the new node instance; it's how we
    # implement copying reference cycles
    transmogrified_nodes: dict[int, _SlotTreeNode] = {
        id(src_tree): copied_tree}
    copy_stack: list[_SlotTreeTraversalFrame] = [_SlotTreeTraversalFrame(
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
        next_slot_name, next_subtree = next_slot_route

        next_subtree_id = id(next_subtree)
        already_copied_node = transmogrified_nodes.get(next_subtree_id)
        # This could be either the first time we hit a recursive subtree,
        # or a non-recursive subtree.
        if already_copied_node is None:
            new_subtree = _SlotTreeNode(
                is_recursive=next_subtree.is_recursive,
                is_terminus=next_subtree.is_terminus)

            # We only need to do this for recursive subtrees; that's the only
            # time we need to get a reference back to a previous subtree.
            if next_subtree.is_recursive:
                transmogrified_nodes[next_subtree_id] = new_subtree

            current_stack_frame.existing_subtree.append(
                _SlotTreeRoute(
                    next_slot_name,
                    new_subtree,
                    slot_types=next_slot_route.slot_types))
            copy_stack.append(_SlotTreeTraversalFrame(
                next_subtree_index=0,
                existing_subtree=new_subtree,
                insertion_subtree=next_subtree))

        else:
            current_stack_frame.existing_subtree.append(
                _SlotTreeRoute(
                    next_slot_name,
                    already_copied_node,
                    slot_types=next_slot_route.slot_types))

        current_stack_frame.next_subtree_index += 1

    return copied_tree, transmogrified_nodes


def _merge_into_slot_tree(
        existing_tree: _SlotTreeNode,
        root_slot_name: str,
        to_merge_slot_type:
            type[TemplateParamsInstance] | ForwardReferenceParam,
        to_merge: _SlotTreeNode
        ) -> dict[int, _SlotTreeNode]:
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

    We return a lookup of ``{id(source_node): dest_node}``, which can
    be used for recording and/or resolving pending forward refs.
    """
    all_transmogrified_nodes: dict[int, _SlotTreeNode] = {}

    # Counterintuitive: since were MERGING trees, the existing_subtree is
    # actually the DESTINATION, and the insertion_subtree the source!
    merge_stack: list[_SlotTreeTraversalFrame] = [_SlotTreeTraversalFrame(
        next_subtree_index=0,
        existing_subtree=existing_tree,
        insertion_subtree=_SlotTreeNode([
            _SlotTreeRoute.from_slot_type(
                root_slot_name,
                to_merge,
                slot_type=to_merge_slot_type)]))]
    # Yes, in theory, this one specific operation of merging trees would be
    # faster if the trees were dicts instead of iterative structures. But
    # we're not optimizing for tree merging; we're optimizing for rendering!
    # And in that case, we're better off with a simple iterative structure.
    while merge_stack:
        current_stack_frame = merge_stack[-1]
        if current_stack_frame.exhausted:
            merge_stack.pop()
            continue

        next_slot_route = current_stack_frame.insertion_subtree[
            current_stack_frame.next_subtree_index]
        next_slot_name, next_subtree = next_slot_route

        if current_stack_frame.existing_subtree.has_route_for(next_slot_name):
            next_existing_route = (
                current_stack_frame.existing_subtree.get_route_for(
                    next_slot_name))
            __, next_existing_subtree = next_existing_route

            # We haven't hit a leaf node yet, so we need to keep looking
            # deeper.
            if next_subtree:
                merge_stack.append(_SlotTreeTraversalFrame(
                    next_subtree_index=0,
                    existing_subtree=next_existing_subtree,
                    insertion_subtree=next_subtree))

            # Note that both of the remaining cases indicate that
            # we hit a leaf node on the to-merge tree. Note though that
            # this isn't definitively also a leaf node on the existing
            # tree, in case you have a union between two very similar, but
            # not identical, trees. Hence the two cases.
            elif next_existing_subtree:
                # TODO: we actually have most of the puzzle pieces for this
                # already in place, we just need to wire up the terminus
                raise NotImplementedError('''
                    You've hit a very particular edge case that we haven't
                    implemented yet.

                    Specifically, you have two very similar set of template
                    slot trees, but they differ slightly. One of them has
                    a "terminus" node at a particular point -- ie, a leaf
                    node for that particular template slot class -- while
                    the other continues on deeper within the slot tree
                    **to the same template slot class**.

                    We really need to update the slot tree to have
                    dedicated leaf node objects, but until then, we can't
                    accommodate this edge case without introducing bugs.

                    If you encounter this, please search for and/or file
                    an issue on github!
                    ''')
                copied_leaf: _SlotTreeNode = []
                current_stack_frame.existing_subtree.append(
                    (next_slot_name, copied_leaf))
                all_transmogrified_nodes[id(next_subtree)] = copied_leaf

            # In this case, it really IS identical, but we've nothing to do.
            else:
                pass

            next_existing_subtree.is_recursive |= next_subtree.is_recursive
            next_existing_subtree.is_terminus |= next_subtree.is_terminus
            if next_existing_subtree.is_recursive:
                all_transmogrified_nodes[id(next_subtree)] = (
                    next_existing_subtree)
            next_existing_route.slot_types.update(next_slot_route.slot_types)

        # Note because it's almost above the fold: this is the "we don't have
        # an existing route for the next_slot_name" case.
        else:
            copied_tree, copied_node_lookup = _copy_slot_tree(next_subtree)
            current_stack_frame.existing_subtree.append(
                # Just as a reminder, this is deepening the tree by 1 level
                _SlotTreeRoute(
                    next_slot_name,
                    copied_tree,
                    slot_types=next_slot_route.slot_types))
            all_transmogrified_nodes.update(copied_node_lookup)

        current_stack_frame.next_subtree_index += 1

    return all_transmogrified_nodes


@dataclass(slots=True)
class _SlotTreeTraversalFrame:
    next_subtree_index: int
    existing_subtree: _SlotTreeNode
    insertion_subtree: _SlotTreeNode

    @property
    def exhausted(self) -> bool:
        """Returns True if the to_merge_subtree has been exhausted, and
        there are no more subtrees to merge.
        """
        return self.next_subtree_index >= len(self.insertion_subtree)


# Note: mutablemapping because otherwise chainmap complains. Even though they
# aren't actually implemented, this is a quick way of getting typing to work
@dataclass(kw_only=True, slots=True)
class _TypehintForwardrefLookup(MutableMapping[str, type]):
    template_module: str
    template_scope: FrameType | None
    captured_refs: set[ForwardReferenceParam] = field(default_factory=set)

    def __getitem__(self, key: str) -> type:
        forward_ref = ForwardReferenceParam(
            module=self.template_module,
            name=key,
            scope=self.template_scope)

        class ForwardReferenceProxyClass:
            """When we return a forward reference, we want to retain all
            of the expected behavior with types -- unions via ``|``,
            etc -- and therefore, we want to return a proxy class
            instead of the forward reference itself.
            """
            REFERENCE_TARGET = forward_ref

        self.captured_refs.add(forward_ref)
        return ForwardReferenceProxyClass

    def __iter__(self) -> Iterator[str]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __setitem__(self, key, value) -> None:
        raise NotImplementedError

    def __delitem__(self, key) -> None:
        raise NotImplementedError


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
            ``ForwardReferenceParam``, which will then hopefully be
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
        template_module = cls.__module__
        forwardref_lookup = _TypehintForwardrefLookup(
            template_module=template_module,
            template_scope=_extract_frame_scope())
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
        for_cls=cls,
        slots=slots,
        vars_=vars_,
        content=content)
    converter_cls = namedtuple('TemplateyConverters', tuple(prerenderers))
    cls._templatey_prerenderers = converter_cls(**prerenderers)
    _resolve_forward_references(cls)
    return cls


def _resolve_forward_references(
        pending_template_cls: type[TemplateIntersectable]):
    """The very last thing to do before we return the class after
    template decoration is to resolve all forward references inside the
    class. To do that, we first need to construct the corresponding
    ForwardReferenceParam and check for it in the pending forward refs
    lookup.

    If we find one, we then need to update the values there, while
    checking for and correctly handling recursion.
    """
    lookup_key = ForwardReferenceParam(
        module=pending_template_cls.__module__,
        name=pending_template_cls.__name__,
        scope=_extract_frame_scope())
    print(lookup_key)
    # raise NotImplementedError(
    #     '''
        
        
    #     TODO LEFT OFF HERE
        
        
    #     we're not getting the correct frames, because the proxy we're using
    #     to construct them is pulling the frame from stdlib collections instead
    #     of the caller library. oops.
        
    #     otherwise, basically, you just need to finish what's described
    #     in the docstring.
        
    #     the other option would be to ... shit, I'm not even sure, tbh.
    #     the whole tree copying business would have to be moved, I think?
    #     if you wanted to do this as a "on first render" kind of thing?
    #     plus that would make the render driver really ugly.

    #     maybe it would be an "on first load"? that would at least be better
    #     than in the render driver.
        
    #     but still... really not sure how to approach this if I can't get the
    #     frame scope to match up.
        
        
    #     ''')


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
