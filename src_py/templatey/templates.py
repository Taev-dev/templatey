from __future__ import annotations

import functools
import inspect
import logging
import sys
import typing
from collections import ChainMap
from collections import namedtuple
from collections.abc import Callable
from collections.abc import Collection
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import _MISSING_TYPE
from dataclasses import Field
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from textwrap import dedent
from types import FrameType
from typing import Annotated
from typing import Any
from typing import Literal
from typing import Protocol
from typing import dataclass_transform
from typing import overload

from docnote import ClcNote

from templatey._annotations import InterfaceAnnotationFlavor
from templatey._forwardrefs import ForwardRefGeneratingNamespaceLookup
from templatey._forwardrefs import ForwardRefLookupKey
from templatey._forwardrefs import extract_frame_scope_id
from templatey._forwardrefs import resolve_forward_references
from templatey._signature import TemplateSignature
from templatey._types import Content
from templatey._types import DynamicClassSlot
from templatey._types import Slot
from templatey._types import Var
from templatey.interpolators import NamedInterpolator
from templatey.parser import InterpolationConfig

logger = logging.getLogger(__name__)


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


# Yes, this is awkward with a bazillion return statements, but in this case,
# clarity is better than elegance
def _classify_interface_field_flavor(  # noqa: PLR0911
        parent_class_type_hints: dict[str, Any],
        template_field: Field
        ) -> tuple[set[InterfaceAnnotationFlavor], type | None] | None:
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
        return {InterfaceAnnotationFlavor.VARIABLE}, nested_type
    elif anno_origin is Slot:
        nested_type, = typing.get_args(resolved_field_type)
        return {InterfaceAnnotationFlavor.SLOT}, nested_type
    elif anno_origin is Content:
        nested_type, = typing.get_args(resolved_field_type)
        return {InterfaceAnnotationFlavor.CONTENT}, nested_type
    elif anno_origin is DynamicClassSlot:
        return {
            InterfaceAnnotationFlavor.SLOT,
            InterfaceAnnotationFlavor.DYNAMIC
        }, None

    # This is all if there's no parameter passed, ex ``foo: Var`` or
    # ``bar: Slot``
    elif anno_origin is None:
        if resolved_field_type is Var:
            return {InterfaceAnnotationFlavor.VARIABLE}, None

        elif resolved_field_type is Content:
            return {InterfaceAnnotationFlavor.CONTENT}, None

        elif resolved_field_type is DynamicClassSlot:
            return {
                InterfaceAnnotationFlavor.SLOT,
                InterfaceAnnotationFlavor.DYNAMIC
            }, None

        elif resolved_field_type is Slot:
            raise TypeError(
                '``Slot`` annotations require a concrete slot class as a '
                + 'parameter!')

        else:
            return None

    # So there was a parameter passed, but the generic wasn't one of ours.
    # Therefore, we can't do anything with it.
    else:
        return None


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
    template_scope_id = extract_frame_scope_id()
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

        # If you're confused, keep reading the comments below.
        forwardref_lookup = ForwardRefGeneratingNamespaceLookup(
            template_module=template_module,
            template_scope_id=template_scope_id)
        # This is the same as the current implementation of get_type_hints
        # in cpython for classes:
        # https://github.com/python/cpython/blob/0045100ccbc3919e8990fa59bc413fe38d21b075/Lib/typing.py#L2325
        template_globals = getattr(
            sys.modules.get(template_module, None), '__dict__', {})

        # Okay, so... this is non-intuitive, to be sure. And if you fail to
        # do this, you'll have really perfidious bugs, because anything that
        # was a valid reference within the temporary namespace created inside
        # the class creation -- **including the class itself** -- will be
        # converted to a forward reference proxy. This would of course cause
        # problems with inter-class references, but it also means that the
        # class itself gets treated as a forward ref instead a concrete slot,
        # and therefore added into the pending forward refs registration. This
        # then gets immediately fixed, except, because the registration is a
        # set, and the iteration order of sets is non-deterministic, the
        # exact, arbitrary order of pending ref resolution can cause a
        # correctly-resolved forward ref for the current class being processed,
        # to then be overwritten by a pending reference, which is then never
        # resolved. Believe me, it's a mess.
        # So to solve this, we need to do two things. First, add the class'
        # __dict__ into the namespace lookups, because that solves any
        # inter-class references. But second, we need to also add the class
        # itself, under its __name__, because this name, though valid and
        # accessible in the temporary class creation namespace, is nonetheless
        # removed from the __dict__ during final class creation.
        # But, ONE FINAL THING: chainmaps need **mutable** mappings. So we
        # have to create a copy of the cls.__dict__ anyways.
        resurrected_clsdef_namespace = {**cls.__dict__, cls.__name__: cls}

        # Note: the whole point of this is to support forward refs, and there's
        # a method to the madness here. We need a way of getting an actual
        # forward ref object. Prior to 3.14 (via PEP 649), we don't have a way
        # to specify to ``get_type_hints`` that we want to have a forward ref
        # object, and it just raises a nameerror that we can't recover from.
        # Our workaround is to inject an object that will lazily create proxy
        # objects for all forward references, but we have to supply this to
        # get_type_hints somehow, which is non-trivial:
        # globalns needs to be strictly a dict, because it gets delegated into
        # ``eval``, which requires one. Which means we can only use the localns
        # to intercept missing forward references. But that, then, means that
        # we need to recover the existing check for the actual globals, since
        # otherwise **all** global names would be overwritten by the forward
        # reference.
        maybe_locals = _extract_template_class_locals()
        if maybe_locals is None:
            prioritized_lookups = (
                # See note above about this
                resurrected_clsdef_namespace,
                template_globals,
                # Fun fact: these aren't included in the other globals!
                __builtins__,
                forwardref_lookup)

        else:
            prioritized_lookups = (
                # See note above about this
                resurrected_clsdef_namespace,
                maybe_locals,
                template_globals,
                # Fun fact: these aren't included in the other globals!
                __builtins__,
                forwardref_lookup)

        # Because of our forward lookup, this will always succeed
        template_type_hints = typing.get_type_hints(
            cls, localns=ChainMap(*prioritized_lookups))

    slots = {}
    dynamic_slots = {}
    vars_ = {}
    content = {}
    prerenderers = {}
    # Note that this ignores initvars, which is what we want
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
            field_flavors, wrapped_type = field_classification

            # A little awkward to effectively just repeat the comparison we did
            # when classifying, but that makes testing easier and control flow
            # clearer
            if InterfaceAnnotationFlavor.VARIABLE in field_flavors:
                dest_lookup = vars_
            elif InterfaceAnnotationFlavor.SLOT in field_flavors:
                if InterfaceAnnotationFlavor.DYNAMIC in field_flavors:
                    dest_lookup = dynamic_slots
                else:
                    dest_lookup = slots
            else:
                dest_lookup = content

            dest_lookup[template_field.name] = wrapped_type
            prerenderers[template_field.name] = template_field.metadata.get(
                'templatey.prerenderer')

    cls._templatey_signature = TemplateSignature.new(
        template_cls=cls,
        slots=slots,
        dynamic_class_slot_names=set(dynamic_slots),
        vars_=vars_,
        content=content,
        forward_ref_lookup_key=template_forward_ref)
    converter_cls = namedtuple('TemplateyConverters', tuple(prerenderers))
    cls._templatey_prerenderers = converter_cls(**prerenderers)

    # Note: this needs to be the absolute last thing, because we need to fully
    # satisfy the intersectable interface before we can call it.
    resolve_forward_references(cls)
    return cls


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
