from __future__ import annotations

import itertools
import logging
from collections import deque
from collections.abc import Callable
from collections.abc import Collection
from collections.abc import Hashable
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from functools import singledispatch
from typing import cast
from typing import overload

from templatey._annotations import InterfaceAnnotationFlavor
from templatey._bootstrapping import EMPTY_TEMPLATE_INSTANCE
from templatey._bootstrapping import EMPTY_TEMPLATE_XABLE
from templatey.exceptions import MismatchedTemplateSignature
from templatey.exceptions import TemplateFunctionFailure
from templatey.parser import InterpolatedContent
from templatey.parser import InterpolatedFunctionCall
from templatey.parser import InterpolatedSlot
from templatey.parser import InterpolatedVariable
from templatey.parser import NestedContentReference
from templatey.parser import NestedVariableReference
from templatey.parser import ParsedTemplateResource
from templatey.templates import ComplexContent
from templatey.templates import EnvFuncInvocation
from templatey.templates import InjectedValue
from templatey.templates import TemplateClass
from templatey.templates import TemplateConfig
from templatey.templates import TemplateIntersectable
from templatey.templates import TemplateParamsInstance
from templatey.templates import TemplateProvenance
from templatey.templates import TemplateProvenanceNode
from templatey.templates import TemplateSignature
from templatey.templates import is_template_instance

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FuncExecutionRequest:
    name: str
    args: Iterable[object]
    kwargs: Mapping[str, object]
    result_key: _PrecallCacheKey


@dataclass(frozen=True)
class FuncExecutionResult:
    # Note: must match signature from TemplateFunction!
    name: str
    retval: Sequence[str | TemplateParamsInstance | InjectedValue] | None
    exc: Exception | None

    def filter_injectables(self) -> Iterable[TemplateParamsInstance]:
        if self.retval is not None:
            for item in self.retval:
                if is_template_instance(item):
                    # Hmm, somehow the TypeIs isn't working
                    yield item  # type: ignore


@dataclass(slots=True)
class RenderEnvRequest:
    to_load: Collection[type[TemplateParamsInstance]]
    to_execute: Collection[FuncExecutionRequest]
    error_collector: list[Exception]

    # These store results; we're adding them inplace instead of needing to
    # merge them later on
    results_loaded: dict[type[TemplateParamsInstance], ParsedTemplateResource]
    results_executed: dict[_PrecallCacheKey, FuncExecutionResult]


def render_driver(
        template_instance: TemplateParamsInstance,
        output: list[str],
        error_collector: list[Exception]
        ) -> Iterable[RenderEnvRequest]:
    """This is a shared method for driving rendering, used by both async
    and sync renderers. It mutates the output list inplace, and yields
    back batched requests for the render environment.
    """
    context = _RenderContext(
        template_preload={},
        function_precall={},
        error_collector=error_collector)
    yield from context.prepopulate(template_instance)
    render_stack: list[_RenderStackNode] = []

    template_xable = cast(TemplateIntersectable, template_instance)
    render_stack.append(
        _RenderStackNode(
            parts=iter(
                context.template_preload[type(template_instance)].parts),
            config=template_xable._templatey_config,
            signature=template_xable._templatey_signature,
            provenance=TemplateProvenance((
                TemplateProvenanceNode(
                    parent_slot_key='',
                    parent_slot_index=-1,
                    instance_id=id(template_instance),
                    instance=template_instance),)),
            instance=template_instance))

    while render_stack:
        try:
            render_node = render_stack[-1]
            next_part = next(render_node.parts)

            # Strings are hardest to deal with because they're containers, so
            # just get that out of the way first
            if isinstance(next_part, str):
                output.append(next_part)

            elif isinstance(next_part, InterpolatedVariable):
                unescaped_vars = _ParamLookup(
                    template_provenance=render_node.provenance,
                    template_preload=context.template_preload,
                    param_flavor=InterfaceAnnotationFlavor.VARIABLE,
                    error_collector=error_collector,
                    placeholder_on_error='')
                unescaped_val = _apply_format(
                    unescaped_vars[next_part.name],
                    next_part.conversion,
                    next_part.format_spec)
                output.append(
                    render_node.config.variable_escaper(unescaped_val))

            elif isinstance(next_part, InterpolatedContent):
                unverified_content = _ParamLookup(
                    template_provenance=render_node.provenance,
                    template_preload=context.template_preload,
                    param_flavor=InterfaceAnnotationFlavor.CONTENT,
                    error_collector=error_collector,
                    placeholder_on_error='')
                val_from_params = unverified_content[next_part.name]

                if isinstance(val_from_params, str):
                    render_node.config.content_verifier(val_from_params)
                    output.append(val_from_params)

                else:
                    output.extend(_render_complex_content(
                        val_from_params,
                        _ParamLookup(
                            template_provenance=render_node.provenance,
                            template_preload=context.template_preload,
                            param_flavor=InterfaceAnnotationFlavor.VARIABLE,
                            error_collector=error_collector,
                            placeholder_on_error=''),
                        render_node.config,
                        error_collector,
                        parent_part_index=next_part.part_index))

            elif isinstance(next_part, InterpolatedSlot):
                provenance_counter = itertools.count()
                render_stack.extend(reversed(tuple(
                    _RenderStackNode(
                        instance=slot_instance,
                        parts=iter(
                            context.template_preload[
                                type(slot_instance)].parts),
                        config=slot_instance._templatey_config,
                        signature=slot_instance._templatey_signature,
                        provenance=TemplateProvenance(
                            (*render_node.provenance, TemplateProvenanceNode(
                                parent_slot_key=next_part.name,
                                parent_slot_index=next(provenance_counter),
                                instance_id=id(slot_instance),
                                instance=slot_instance))))
                    for slot_instance
                    in getattr(render_node.instance, next_part.name))))

            elif isinstance(next_part, InterpolatedFunctionCall):
                execution_result = context.function_precall[
                    _get_precall_cache_key(render_node.provenance, next_part)]
                nested_render_nodes = _build_render_node_for_func_result(
                    execution_result,
                    render_node.config,
                    error_collector,
                    context.template_preload)
                render_stack.extend(nested_render_nodes)

            else:
                raise TypeError(
                    'impossible branch: invalid template part type!')

        except StopIteration:
            render_stack.pop()

        # Note: this could be eg a lookup error because of a missing variable.
        # This isn't redundant with the error collection within prepopulation.
        except Exception as exc:
            error_collector.append(exc)


@dataclass(slots=True)
class _RenderStackNode:
    instance: TemplateParamsInstance
    parts: Iterator[
        str
        | InterpolatedSlot
        | InterpolatedContent
        | InterpolatedVariable
        | InterpolatedFunctionCall]
    config: TemplateConfig
    signature: TemplateSignature
    provenance: TemplateProvenance


@dataclass(slots=True)
class _RenderContext:
    template_preload: dict[TemplateClass, ParsedTemplateResource]
    function_precall: dict[_PrecallCacheKey, FuncExecutionResult]
    error_collector: list[Exception]

    def prepopulate(
            self,
            root_template: TemplateParamsInstance
            ) -> Iterable[RenderEnvRequest]:
        """For the passed root template, populates the template_preload
        and function_precall until either all resources have been
        prepared, or it needs help from the render environment.
        """
        # Note that there will only be multiple local roots on any given
        # loop iteration if a function returns multiple template instances
        # as a result.
        template_backlog_local_roots: list[TemplateParamsInstance]
        template_backlog_included_classes: set[TemplateClass]

        function_backlog: list[EnvFuncInvocation]
        template_preload: dict[TemplateClass, ParsedTemplateResource]
        function_precall: dict[_PrecallCacheKey, FuncExecutionResult]

        root_template_class = type(root_template)
        root_template_xable = cast(TemplateIntersectable, root_template_class)
        # Need to make a copy since we'll be mutating, but we don't need to
        # check for membership, so we might as well keep it a list
        template_backlog_local_roots = [root_template]
        template_backlog_included_classes = set(
            root_template_xable._templatey_signature.included_template_classes)
        template_backlog_included_classes.add(type(root_template))
        function_backlog = []
        template_preload = self.template_preload
        function_precall = self.function_precall

        # Note: it might seem a little redundant that we're looping over this
        # and then iterating across all of the template instances; however,
        # this allows us to batch together all of the help requests, which is
        # a huge reduction in runtime overhead
        while bool(template_backlog_local_roots) or bool(function_backlog):
            to_execute: list[FuncExecutionRequest] = []

            # Note that this will never run on the first iteration of the outer
            # loop, because we won't have a function backlog until after the
            # templates have been loaded.
            while function_backlog:
                template_provenance, function_call = function_backlog.pop()
                unescaped_vars = _ParamLookup(
                    template_provenance=template_provenance,
                    template_preload=template_preload,
                    param_flavor=InterfaceAnnotationFlavor.VARIABLE,
                    error_collector=self.error_collector,
                    placeholder_on_error='')
                unverified_content = _ParamLookup(
                    template_provenance=template_provenance,
                    template_preload=template_preload,
                    param_flavor=InterfaceAnnotationFlavor.CONTENT,
                    error_collector=self.error_collector,
                    placeholder_on_error='')

                # Note that the full call signature is **defined** within
                # the parsed template body, but it may **reference** vars
                # and/or content within the template instance.
                args = _recursively_coerce_func_execution_params(
                    function_call.call_args,
                    unescaped_vars=unescaped_vars,
                    unverified_content=unverified_content)
                kwargs = _recursively_coerce_func_execution_params(
                    function_call.call_kwargs,
                    unescaped_vars=unescaped_vars,
                    unverified_content=unverified_content)

                result_cache_key = _get_precall_cache_key(
                    template_provenance, function_call)
                to_execute.append(
                    FuncExecutionRequest(
                        function_call.name,
                        args=args,
                        kwargs=kwargs,
                        result_key=result_cache_key))

            yield RenderEnvRequest(
                to_load=template_backlog_included_classes,
                to_execute=to_execute,
                error_collector=self.error_collector,
                results_loaded=template_preload,
                results_executed=function_precall)

            # Always do this AFTER the env request has been sent; otherwise,
            # we can't possibly add any more functions to the backlog, because
            # we won't know anything more about the templates!
            while template_backlog_local_roots:
                local_root_template = template_backlog_local_roots.pop()
                local_root_template_signature = cast(
                    TemplateIntersectable,
                    local_root_template)._templatey_signature

                function_backlog.extend(
                    local_root_template_signature.extract_function_invocations(
                        local_root_template, template_preload))
            template_backlog_included_classes.clear()

            # Now that we've cleared any existing templates from the backlog,
            # we need to add in any that were created by the functions.
            # Note that it's possible that the function backlog could return
            # a new template instance of an already loaded template class. In
            # that case, we still need to execute the functions there, so the
            # easiest thing is just to ask for a reload of the template.
            while to_execute:
                result_cache_key = to_execute.pop().result_key
                function_result = function_precall[result_cache_key]

                for injected_template in function_result.filter_injectables():
                    injected_xable = cast(
                        TemplateIntersectable, injected_template)
                    template_backlog_local_roots.append(injected_template)
                    template_backlog_included_classes.update(
                        injected_xable._templatey_signature
                        .included_template_classes)
                    template_backlog_included_classes.add(
                        type(injected_template))


type _PrecallExecutionRequest = tuple[
    TemplateProvenance, InterpolatedFunctionCall]
type _PrecallCacheKey = Hashable


def _get_precall_cache_key(
        template_provenance: TemplateProvenance,
        interpolated_call: InterpolatedFunctionCall
        ) -> _PrecallCacheKey:
    """For a particular template instance and interpolated function
    call, creates the hashable cache key to be used for the render
    context.
    """
    # Note that template provenance includes the actual current template
    # instance, so by definition, this encodes exactly which function call
    # is being referenced
    return (template_provenance, interpolated_call)


def _render_complex_content(
        complex_content: ComplexContent | object,
        unescaped_vars: _ParamLookup,
        template_config: TemplateConfig,
        error_collector: list[Exception],
        *,
        parent_part_index: int,
        ) -> Iterable[str]:
    # Extra typecheck here because we're calling in to this potentially from
    # an unverified context
    if isinstance(complex_content, ComplexContent):
        for content_segment in complex_content.flatten(
            unescaped_vars,
            parent_part_index
        ):
            if isinstance(content_segment, str):
                template_config.content_verifier(content_segment)
                yield content_segment

            elif isinstance(content_segment, InterpolatedVariable):
                unescaped_val = _apply_format(
                    # Note that we've already checked this is defined in
                    # _flatten_and_interpolate, so we don't need .get()
                    unescaped_vars[content_segment.name],
                    content_segment.conversion,
                    content_segment.format_spec)
                yield template_config.variable_escaper(unescaped_val)

            else:
                error_collector.append(_capture_traceback(TypeError(
                    'ComplexContent.flatten() must always return strings '
                    + 'or InterpolatedVariable instances!',
                    content_segment)))

    else:
        error_collector.append(_capture_traceback(TypeError(
            'Interpolated content values must always be strings or '
            + 'ComplexContent instances!', complex_content)))


def _build_render_node_for_func_result(
        execution_result: FuncExecutionResult,
        template_config: TemplateConfig,
        error_collector: list[Exception],
        template_preload: dict[TemplateClass, ParsedTemplateResource]
        ) -> Iterable[_RenderStackNode]:
    """This constructs a _RenderNode for the given execution result and
    returns it (or None, if there was an error).
    """
    has_nested_templates = False
    resulting_parts: list[str | TemplateParamsInstance] = []
    if execution_result.exc is None:
        if execution_result.retval is None:
            raise TypeError(
                'Impossible branch! Malformed func exe result',
                execution_result)

        for result_part in execution_result.retval:
            if isinstance(result_part, str):
                resulting_parts.append(
                    template_config.variable_escaper(result_part))
            elif isinstance(result_part, InjectedValue):
                resulting_parts.append(
                    _coerce_injected_value(result_part, template_config))
            elif is_template_instance(result_part):
                has_nested_templates = True
                resulting_parts.append(result_part)
            else:
                error_collector.append(_capture_traceback(
                    TypeError(
                        'Invalid return from env function!',
                        execution_result, result_part)))

    else:
        if execution_result.retval is not None:
            raise TypeError(
                'Impossible branch! Malformed func exe result',
                execution_result)

        error_collector.append(_capture_traceback(
            TemplateFunctionFailure('Env function raised!'),
            from_exc=execution_result.exc))

    if resulting_parts:
        if has_nested_templates:
            return _build_render_stack_extension(
                resulting_parts,
                template_config,
                template_preload)

        # This is an optimization that skips a bunch of copies and stack
        # reordering if the function didn't have any nested templates
        else:
            return [_RenderStackNode(
                parts=iter(cast(list[str], resulting_parts)),
                config=EMPTY_TEMPLATE_XABLE._templatey_config,
                signature=EMPTY_TEMPLATE_XABLE._templatey_signature,
                provenance=TemplateProvenance(),
                instance=EMPTY_TEMPLATE_INSTANCE)]

    return ()


def _build_render_stack_extension(
        func_result_parts: list[str | TemplateParamsInstance],
        parent_template_config: TemplateConfig,
        template_preload: dict[TemplateClass, ParsedTemplateResource]
        ) -> deque[_RenderStackNode]:
    """Our render stack needs to be structured in a particular way, but
    the results from a function execution may interleave plain strings
    with nested templates. This handles converting the parts into a
    flattened list of stack nodes that preserves part order while still
    maintaining provenance etc.
    """
    nodes: deque[_RenderStackNode] = deque()
    current_node_parts: list[str] = []

    for func_result_part in func_result_parts:
        if isinstance(func_result_part, str):
            current_node_parts.append(func_result_part)

        else:
            if current_node_parts:
                nodes.append(_RenderStackNode(
                    parts=iter(current_node_parts),
                    config=EMPTY_TEMPLATE_XABLE._templatey_config,
                    signature=EMPTY_TEMPLATE_XABLE._templatey_signature,
                    provenance=TemplateProvenance(),
                    instance=EMPTY_TEMPLATE_INSTANCE))
                # Make sure to create a new one so we don't screw up our iter()
                current_node_parts = []

            template_xable = cast(TemplateIntersectable, func_result_part)
            nodes.append(
                _RenderStackNode(
                    parts=iter(
                        template_preload[type(func_result_part)].parts),
                    config=template_xable._templatey_config,
                    signature=template_xable._templatey_signature,
                    provenance=TemplateProvenance((
                        TemplateProvenanceNode(
                            parent_slot_key='',
                            parent_slot_index=-1,
                            instance_id=id(func_result_part),
                            instance=func_result_part),)),
                    instance=func_result_part))

    # We have to do this one last time in case there were any trailing strings
    # after the last nested template instance
    if current_node_parts:
        nodes.append(_RenderStackNode(
            parts=iter(current_node_parts),
            config=EMPTY_TEMPLATE_XABLE._templatey_config,
            signature=EMPTY_TEMPLATE_XABLE._templatey_signature,
            provenance=TemplateProvenance(),
            instance=EMPTY_TEMPLATE_INSTANCE))
    # Rationale: the nodes are currently in order of first encountered to
    # last encountered, which is the opposite of a stack. By simply reversing
    # them, we can then just extend them onto the existing render stack, easy
    # peasy.
    nodes.reverse()
    return nodes


# Note: it would be nice if we could get a little more clever with the types
# on this, but having the lookup be passed in as a callable makes it pretty
# awkward
@dataclass(slots=True, init=False)
class _ParamLookup(Mapping[str, object]):
    """This is a highly-performant layer of indirection that avoids most
    dictionary copies, but nonetheless allows us to both have helpful
    error messages, and collect all possible errors into a single
    ExceptionGroup (without short-circuiting on the first error) while
    rendering.
    """
    template_provenance: TemplateProvenance
    error_collector: list[Exception]
    placeholder_on_error: object
    lookup: Callable[[str], object]
    param_flavor: InterfaceAnnotationFlavor

    def __init__(
            self,
            template_provenance: TemplateProvenance,
            template_preload: dict[TemplateClass, ParsedTemplateResource],
            param_flavor: InterfaceAnnotationFlavor,
            error_collector: list[Exception],
            placeholder_on_error: object):
        self.error_collector = error_collector
        self.placeholder_on_error = placeholder_on_error
        self.template_provenance = template_provenance
        self.param_flavor = param_flavor

        if param_flavor is InterfaceAnnotationFlavor.CONTENT:
            self.lookup = partial(
                template_provenance.bind_content,
                template_preload=template_preload)
        elif param_flavor is InterfaceAnnotationFlavor.VARIABLE:
            self.lookup = partial(
                template_provenance.bind_variable,
                template_preload=template_preload)
        else:
            raise TypeError(
                'Internal templatey error: _ParamLookup not supported with '
                + 'that flavor', param_flavor)

    def __getitem__(self, name: str) -> object:
        try:
            return self.lookup(name)

        except KeyError as exc:
            self.error_collector.append(_capture_traceback(
                MismatchedTemplateSignature(
                    'Template referenced invalid param in a way that was not '
                    + 'caught during template loading. This could indicate '
                    + 'referencing eg a slot as content, content as var, etc. '
                    + 'Or it could indicate an ellipsis being passed in as '
                    + 'the value for a template parameter. Or it could be a '
                    + 'bug in templatey.',
                    self.template_provenance[-1].instance,
                    name),
                from_exc=exc))
            return self.placeholder_on_error

    def __len__(self) -> int:
        # Note: this is going to be less commonly used (presumably) than just
        # getitem (the only external access to this is through complex content
        # flattening), so don't precalculate this during __init__
        template_instance = self.template_provenance[-1].instance
        template_xable = cast(TemplateIntersectable, template_instance)
        if self.param_flavor is InterfaceAnnotationFlavor.CONTENT:
            return len(template_xable._templatey_signature.content_names)
        elif self.param_flavor is InterfaceAnnotationFlavor.VARIABLE:
            return len(template_xable._templatey_signature.var_names)
        else:
            raise TypeError(
                'Internal templatey error: _ParamLookup not supported with '
                + 'that flavor', self.param_flavor)

    def __iter__(self) -> Iterator[str]:
        # Note: this is going to be less commonly used (presumably) than just
        # getitem (the only external access to this is through complex content
        # flattening), so don't precalculate this during __init__
        template_instance = self.template_provenance[-1].instance
        template_xable = cast(TemplateIntersectable, template_instance)
        if self.param_flavor is InterfaceAnnotationFlavor.CONTENT:
            return (
                getattr(template_instance, attr_name)
                for attr_name
                in template_xable._templatey_signature.content_names)
        elif self.param_flavor is InterfaceAnnotationFlavor.VARIABLE:
            return (
                getattr(template_instance, attr_name)
                for attr_name
                in template_xable._templatey_signature.var_names)
        else:
            raise TypeError(
                'Internal templatey error: _ParamLookup not supported with '
                + 'that flavor', self.param_flavor)


@overload
def _recursively_coerce_func_execution_params(
        param_value: str,
        *,
        unescaped_vars: _ParamLookup,
        unverified_content: _ParamLookup
        ) -> str: ...
@overload
def _recursively_coerce_func_execution_params[K: object, V: object](
        param_value: Mapping[K, V],
        *,
        unescaped_vars: _ParamLookup,
        unverified_content: _ParamLookup
        ) -> dict[K, V]: ...
@overload
def _recursively_coerce_func_execution_params[T: object](
        param_value: list[T] | tuple[T],
        *,
        unescaped_vars: _ParamLookup,
        unverified_content: _ParamLookup
        ) -> tuple[T]: ...
@overload
def _recursively_coerce_func_execution_params(
        param_value: NestedContentReference | NestedVariableReference,
        *,
        unescaped_vars: _ParamLookup,
        unverified_content: _ParamLookup
        ) -> object: ...
@overload
def _recursively_coerce_func_execution_params[T: object](
        param_value: T,
        *,
        unescaped_vars: _ParamLookup,
        unverified_content: _ParamLookup
        ) -> T: ...
@singledispatch
def _recursively_coerce_func_execution_params(
        # Note: singledispatch doesn't support type vars
        param_value: object,
        *,
        unescaped_vars: _ParamLookup,
        unverified_content: _ParamLookup
        ) -> object:
    """Templatey templates support references to both content and
    variables as call args/kwargs for environment functions. They also
    support both iterables (lists) and mappings (dicts) as literals
    within the template, each of which can also reference content and
    variables, and might themselves contain iterables or mappings.

    This recursively walks the passed execution params, converting all
    of the content or variable references to their values. If the passed
    value was a container, it creates a new copy of the container with
    the references replaced. Otherwise, it simple returns the passed
    value.

    This, the trivial case, handles any situation where the passed
    param value was a plain object.
    """
    return param_value


# Note: I think there might be a bug in pyright re: singledispatch vs overloads
@_recursively_coerce_func_execution_params.register  # type: ignore
def _(
        # Note: singledispatch doesn't support type vars
        param_value: list | tuple | dict,
        *,
        unescaped_vars: _ParamLookup,
        unverified_content: _ParamLookup
        ) -> tuple | dict:
    """Again, in the container case, we want to create a new copy of
    the container, replacing its values with the recursive call.
    Note that the keys in nested dictionaries cannot be references,
    only the values.
    """
    if isinstance(param_value, dict):
        return {
            contained_key: _recursively_coerce_func_execution_params(
                contained_value,
                unescaped_vars=unescaped_vars,
                unverified_content=unverified_content)
            for contained_key, contained_value in param_value.items()}

    else:
        return tuple(
            _recursively_coerce_func_execution_params(
                contained_value,
                unescaped_vars=unescaped_vars,
                unverified_content=unverified_content)
            for contained_value in param_value)


# Note: I think there might be a bug in pyright re: singledispatch vs overloads
@_recursively_coerce_func_execution_params.register  # type: ignore
def _(
        # Note: singledispatch doesn't support type vars
        param_value: str,
        *,
        unescaped_vars: _ParamLookup,
        unverified_content: _ParamLookup
        ) -> str:
    """We need to be careful here to supply a MORE SPECIFIC dispatch
    type than container for strings, since they are technically also
    containers. Bleh.
    """
    return param_value


# Note: I think there might be a bug in pyright re: singledispatch vs overloads
@_recursively_coerce_func_execution_params.register  # type: ignore
def _(
        param_value: NestedContentReference,
        *,
        unescaped_vars: _ParamLookup,
        unverified_content: _ParamLookup
        ) -> object:
    """Nested content references need to be retrieved from the
    unverified content. Note that this (along with the nested variable
    references) are the whole reason we're doing this execution params
    coercion in the first place.
    """
    return unverified_content[param_value.name]


# Note: I think there might be a bug in pyright re: singledispatch vs overloads
@_recursively_coerce_func_execution_params.register  # type: ignore
def _(
        param_value: NestedVariableReference,
        *,
        unescaped_vars: _ParamLookup,
        unverified_content: _ParamLookup
        ) -> object:
    """Nested variable references need to be retrieved from the
    unescaped vars. Note that this (along with the nested content
    references) are the whole reason we're doing this execution params
    coercion in the first place.
    """
    return unescaped_vars[param_value.name]


def _apply_format(raw_value, conversion, format_spec) -> str:
    """For both interpolated variables and injected values, we allow
    format specs and conversions to be supplied. We need to actually
    apply these, but the stdlib doesn't really give us a good way of
    doing that. So this is how we do that instead.
    """
    # hot path go fast
    if conversion is None and format_spec is None:
        if isinstance(raw_value, str):
            formatted_value = raw_value
        else:
            formatted_value = format(raw_value)

    else:
        # format() expects an empty string, NOT None
        format_spec = format_spec or ''
        if conversion == 's':
            to_format = str(raw_value)
        elif conversion == 'r':
            to_format = repr(raw_value)
        elif conversion:
            raise ValueError('Unknown formatting conversion!', conversion)
        else:
            to_format = raw_value

        formatted_value = format(to_format, format_spec)

    return formatted_value


def _capture_traceback[E: Exception](
        exc: E,
        from_exc: Exception | None = None) -> E:
    """This is a little bit hacky, but it allows us to capture the
    traceback of the exception we want to "raise" but then collect into
    an ExceptionGroup at the end of the rendering cycle. It does pollute
    the traceback with one extra stack level, but the important thing
    is to capture the upstream context for the error, and that it will
    do just fine.

    There's almost certainly a better way of doing this, probably using
    traceback from the stdlib. But this is quicker to code, and that's
    my current priority. Gracefulness can come later!
    """
    try:
        if from_exc is None:
            raise exc
        else:
            raise exc from from_exc

    except type(exc) as exc_with_traceback:
        return exc_with_traceback


def _coerce_injected_value(
        injected_value: InjectedValue,
        template_config: TemplateConfig
        ) -> str:
    """InjectedValue instances are used within the return value of
    environment functions to indicate that the result should be sourced
    from the variables and/or the content of the current render call.
    This function is responsible for converting the InjectedValue
    instance into the final resulting string to render.
    """
    unescaped_value = _apply_format(
        injected_value.value,
        injected_value.conversion,
        injected_value.format_spec)

    if injected_value.use_variable_escaper:
        escapish_value = template_config.variable_escaper(unescaped_value)
    else:
        escapish_value = unescaped_value

    if injected_value.use_content_verifier:
        template_config.content_verifier(escapish_value)

    return escapish_value
