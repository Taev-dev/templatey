from __future__ import annotations

import itertools
import logging
import typing
from collections.abc import Callable
from collections.abc import Collection
from collections.abc import Generator
from collections.abc import Hashable
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from functools import singledispatch
from types import EllipsisType
from typing import cast
from typing import overload

from templatey._annotations import InterfaceAnnotationFlavor
from templatey.exceptions import IncompleteTemplateParams
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
from templatey.templates import InjectedValue
from templatey.templates import TemplateClass
from templatey.templates import TemplateConfig
from templatey.templates import TemplateInstanceID
from templatey.templates import TemplateIntersectable
from templatey.templates import TemplateParamsInstance
from templatey.templates import TemplateProvenance
from templatey.templates import TemplateProvenanceNode
from templatey.templates import TemplateSignature
from templatey.templates import is_template_instance

if typing.TYPE_CHECKING:
    from templatey.environments import RenderEnvironment

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


def render_driver_sync(
        template_instance: TemplateParamsInstance,
        render_environment: RenderEnvironment
        ) -> Iterable[str]:
    """This drives the _flatten_and_interpolate coroutine when
    running in sync mode, and collects all of its errors. More
    details there!
    """
    flattened_or_help_req: TemplateParamsInstance | FuncExecutionRequest | str
    help_response: None | ParsedTemplateResource | FuncExecutionResult = None

    errors = []
    render_recursor = _flatten_and_interpolate(
        template_instance,
        error_collector=errors)
    # Note that we **cannot use a for loop here.** The return value of
    # the call to .send() is the next yield, so if we try to mix together
    # both a ``for flattened_str_or_template_request in ...`` and a
    # ``render_recursor.send(...)``, we skip over flattened strings.
    try:
        while True:
            # Note that it's important that we've pre-initialized the
            # bubbled_request to None, since the first .send() call
            # MUST ALWAYS be None (as per python spec).
            flattened_or_help_req = render_recursor.send(help_response)
            # Always reset this immediately after sending a value, so that
            # we're less likely to accidentally send the same template
            # twice due to some other bug (defense in depth / fail loudly)
            help_response = None
            # The flattening resulted in a string value, ready for final
            # rendering
            if isinstance(flattened_or_help_req, str):
                yield flattened_or_help_req

            elif isinstance(flattened_or_help_req, FuncExecutionRequest):
                help_response = (
                    render_environment.execute_template_function_sync(
                        flattened_or_help_req))

            # The flattening requires a template resource to continue, so
            # we need to load it before continuing.
            else:
                required_template = type(flattened_or_help_req)
                try:
                    help_response = render_environment.load_sync(
                        required_template)
                except Exception as exc:
                    errors.append(exc)

    except StopIteration:
        pass

    if errors:
        raise ExceptionGroup('Failed to render template', errors)


@dataclass(slots=True)
class RenderEnvRequest:
    to_load: Iterable[type[TemplateParamsInstance]]
    to_execute: Iterable[FuncExecutionRequest]
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
                unescaped_val = _apply_format(
                    # Note that we've already checked this is defined in
                    # _flatten_and_interpolate, so we don't need .get()
                    render_node.signature.get_var(
                        render_node.instance, next_part.name),
                    next_part.conversion,
                    next_part.format_spec)
                output.append(
                    render_node.config.variable_escaper(unescaped_val))

            elif isinstance(next_part, InterpolatedContent):
                # Note that we've already checked this is defined in
                # _flatten_and_interpolate, so we don't need .get()
                val_from_params = render_node.signature.get_content(
                    render_node.instance, next_part.name)

                if isinstance(val_from_params, str):
                    render_node.config.content_verifier(val_from_params)
                    output.append(val_from_params)

                else:
                    output.extend(_render_complex_content(
                        val_from_params,
                        render_node.signature.get_all_vars(
                            render_node.instance),
                        render_node.config,
                        error_collector))

            elif isinstance(next_part, InterpolatedSlot):
                slot_class = render_node.signature.slots[next_part.name]
                slot_xable = cast(type[TemplateIntersectable], slot_class)
                provenance_counter = itertools.count()
                render_stack.extend(reversed(tuple(
                    _RenderStackNode(
                        instance=slot_instance,
                        parts=iter(context.template_preload[slot_class].parts),
                        config=slot_xable._templatey_config,
                        signature=slot_xable._templatey_signature,
                        provenance=TemplateProvenance(
                            (*render_node.provenance, TemplateProvenanceNode(
                                parent_slot_key=next_part.name,
                                parent_slot_index=next(provenance_counter),
                                instance_id=id(slot_instance),
                                instance=slot_instance))))
                    for slot_instance
                    in getattr(render_node.instance, next_part.name))))

            elif isinstance(next_part, InterpolatedFunctionCall):
                print('!!!!!')
                print(context.function_precall)
                execution_result = context.function_precall[
                    _get_precall_cache_key(render_node.provenance, next_part)]
                render_node = _build_render_node_for_func_result(
                    execution_result, render_node.config, error_collector)
                if render_node is not None:
                    render_stack.append(render_node)

            else:
                raise TypeError(
                    'impossible branch: invalid template part type!')

        except StopIteration:
            render_stack.pop()

        except Exception as exc:
            error_collector.append(exc)


    return
    raise NotImplementedError(
        '''
        TODO LEFT OFF HERE
        still need to strip out the id->instance lookup
        still need to clean up old unused stuff
        still need to update tests

        the current status is that you have a fully prepopulated
        render context, including the template preload and function
        precall. so you should just be able to start actually rendering.

        you need to make a note somewhere though, that the reason you're
        doing prepopulation separately from the actual rendering is that
        it makes batching feasible:
        ++  when prepopulating, the worst combinatorics are based just on
            the number of slot instances times the number of functions per
            template
        ++  during prepopulating, the worst combinatorics are based on
            the number of template PARTS, times the number of instances
            per slot.
        the reason being that you have to temporarily cache everything
        until you reach the end of the batch.

        the other thing is that in prepopulation, there are clear
        boundaries between batches, which make it relatively easy to
        cache things. but during rendering, I think you'd have to build
        literally the entire render output and cache interim results to
        be able to batch everything, so it's clearly way better to do it
        in two steps instead of one.
        ''')


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
        template_backlog: dict[TemplateClass, list[TemplateProvenance]]
        function_backlog: list[_PrecallExecutionRequest]
        template_preload: dict[TemplateClass, ParsedTemplateResource]
        function_precall: dict[_PrecallCacheKey, FuncExecutionResult]

        root_template_class = cast(TemplateIntersectable, type(root_template))
        template_backlog, __ = (
            root_template_class._templatey_signature
            .apply_slot_tree(root_template))
        function_backlog = []
        template_preload = self.template_preload
        function_precall = self.function_precall

        # Note: it might seem a little redundant that we're looping over this
        # and then iterating across all of the template instances; however,
        # this allows us to batch together all of the help requests, which is
        # a huge reduction in runtime overhead
        while bool(template_backlog) or bool(function_backlog):
            to_load: list[type[TemplateParamsInstance]] = []
            to_execute: list[FuncExecutionRequest] = []
            # These are for keeping track of the things we needed to ask the
            # render env for help on; we have a bit of post-processing to do
            # after the render env loads them.
            to_function_backlog: list[
                tuple[TemplateClass, list[TemplateProvenance]]] = []
            to_template_backlog: list[_PrecallCacheKey] = []

            # Always do the template backlog first -- it's responsible for
            # populating the function backlog!
            while template_backlog:
                (template_class,
                 template_invocations) = template_backlog.popitem()

                # The secondary check here is more for correctness than
                # performance, since it might even be slower than just updating
                # the value. But if we change the caching logic to be bounded
                # size, and if, during loading, another template evicts the
                # current one, our use of ID for cache keys will cause a
                # mismatch in the precall lookup.
                if template_class in template_preload:
                    parsed_template = template_preload[template_class]
                    # The combinatorics here get pretty big, but: we're adding
                    # every combination of (all invocations of that particular
                    # template class) and (all function calls for that template
                    # definition) to the function backlog.
                    function_backlog.extend(itertools.product(
                        template_invocations,
                        itertools.chain.from_iterable(
                            parsed_template.function_calls.values())))

                else:
                    to_load.append(template_class)
                    to_function_backlog.append(
                        (template_class, template_invocations))

            # We've cleared the template backlog, at least for now. Now we need
            # to execute any funtions that were requested, for all of the
            # instances that needed them, and store the results. Note that
            # this might result in loading new template instances, causing
            # another loop of the outermost while loop.
            while function_backlog:
                print('@@@@@')
                template_provenance, function_call = function_backlog.pop()
                print(template_provenance)
                unescaped_vars = _ParamLookup[object](
                    template_provenance=template_provenance,
                    template_preload=template_preload,
                    param_flavor=InterfaceAnnotationFlavor.VARIABLE,
                    error_collector=self.error_collector,
                    placeholder_on_error='')
                unverified_content = _ParamLookup[object](
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

                to_execute.append(
                    FuncExecutionRequest(
                        function_call.name,
                        args=args,
                        kwargs=kwargs,
                        result_key=_get_precall_cache_key(
                            template_provenance, function_call)))

            yield RenderEnvRequest(
                to_load=to_load,
                to_execute=to_execute,
                error_collector=self.error_collector,
                results_loaded=template_preload,
                results_executed=function_precall)

            for template_class, template_invocations in to_function_backlog:
                parsed_template = template_preload[template_class]
                # The combinatorics here get pretty big, but: we're adding
                # every combination of (all invocations of that particular
                # template class) and (all function calls for that template
                # definition) to the function backlog.
                function_backlog.extend(itertools.product(
                    template_invocations,
                    itertools.chain.from_iterable(
                        parsed_template.function_calls.values())))

            for result_cache_key in to_template_backlog:
                function_result = function_precall[result_cache_key]

                for injected_template in function_result.filter_injectables():
                    injected_template_signature = cast(
                        TemplateIntersectable, injected_template
                    )._templatey_signature

                    injected_template_invocations, __ = (
                        injected_template_signature.apply_slot_tree(
                            injected_template))

                    for template_class, template_provenances in (
                        injected_template_invocations
                    ).items():
                        template_backlog[template_class].extend(
                            template_provenances)

            to_function_backlog.clear()
            to_template_backlog.clear()
            to_load.clear()
            to_execute.clear()


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
    return (template_provenance, id(interpolated_call))







def _render_complex_content(
        complex_content: ComplexContent | object,
        unescaped_vars,
        template_config: TemplateConfig,
        error_collector: list[Exception],
        ) -> Iterable[str]:
    # Extra typecheck here because we're calling in to this potentially from
    # an unverified context
    if isinstance(complex_content, ComplexContent):
        for content_segment in complex_content.flatten(unescaped_vars):
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
        error_collector: list[Exception]
        ) -> _RenderStackNode | None:
    """This constructs a _RenderNode for the given execution result and
    returns it (or None, if there was an error).
    """
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
                resulting_parts.append(result_part)
            else:
                error_collector.append(_capture_traceback(
                    TypeError(
                        'Invalid return from template function!',
                        execution_result, result_part)))

    else:
        if execution_result.retval is not None:
            raise TypeError(
                'Impossible branch! Malformed func exe result',
                execution_result)

        error_collector.append(_capture_traceback(
            TemplateFunctionFailure('Template function raised!'),
            from_exc=execution_result.exc))

    if resulting_parts:
        return _RenderStackNode(
            instance=None,
            parts=iter(resulting_parts),
            config=None,
            signature=None,
            provenance=None)


@dataclass(slots=True, init=False)
class _ParamLookup[T]:
    """This is a highly-performant layer of indirection that avoids most
    dictionary copies, but nonetheless allows us to both have helpful
    error messages, and collect all possible errors into a single
    ExceptionGroup (without short-circuiting on the first error) while
    rendering.
    """
    template_provenance: TemplateProvenance
    error_collector: list[Exception]
    placeholder_on_error: T
    lookup: Callable[[str], object]

    def __init__(
            self,
            template_provenance: TemplateProvenance,
            template_preload: dict[TemplateClass, ParsedTemplateResource],
            param_flavor: InterfaceAnnotationFlavor,
            error_collector: list[Exception],
            placeholder_on_error: T):
        self.error_collector = error_collector
        self.placeholder_on_error = placeholder_on_error
        self.template_provenance = template_provenance

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
                    + 'caught during template loading. This likely indicates '
                    + 'referencing eg a slot as content, content as var, etc. '
                    + 'Or it could be a bug in templatey.',
                    self.template_provenance[-1].instance,
                    name),
                from_exc=exc))
            return self.placeholder_on_error


def _flatten_and_interpolate(
        template_instance: TemplateParamsInstance,
        *,
        # We could, in theory, return a list of errors instead, but it's
        # cleaner code if we pass in the error collector as a parameter.
        # It also has the added benefit of avoiding a bunch of .extend()
        # calls because of the recursion.
        error_collector: list[Exception],
        parent_params: dict[str, object] | None = None,
        ) -> Generator[
            TemplateParamsInstance | str | FuncExecutionRequest,
            ParsedTemplateResource | None | FuncExecutionResult,
            None]:
    """Okay, this is gonna get messy, so here's the deal.
    The flattening/interpolation mechanism is messy. We really don't
    want to need to maintain it twice (once for sync, once for
    async). But because it's recursive, and because we don't want
    to recurse over the to-render-tree multiple times, we also need
    to be loading template resources as we go. That presents a
    problem, since the loader could be either sync or async, and
    the call to render could have been either sync or async as well.
    So that means we either need to implement this whole method
    twice (and have clearer code, but a higher maintenance burden),
    or implement it as an old-school coroutine (and have confusing
    code, but only one version of it to maintain). I'm opting for
    the former, because I genuinely think this might actually be
    the route that's less likely to run into problems, as long as
    it's well documented.

    That being said... I'm not sure if that's going to be really
    un-performant, so maybe this ends up being a mistake. There's
    only one way to find out, and that's benchmarking and profiling.

    SO: the way that this works is that we want to flatten out the
    passed template instance, performing any needed interpolations,
    and then recursing into any slots. We're passed in a template
    instance -- ie, the template parameters -- and always request
    a loaded template resource from the caller by yielding it
    back.
    """
    template_xable = cast(TemplateIntersectable, template_instance)
    template_config = template_xable._templatey_config
    all_slots = _ParamLookup[Sequence[TemplateParamsInstance]](
        template_instance=template_instance,
        error_collector=error_collector,
        placeholder_on_error=[],
        valid_param_names=template_xable._templatey_signature.slot_names)

    # TODO: question: is it worth it to also do type checking of the override
    # arguments? This would probably imply something like pydantic if you
    # want to have it be robust, but I'm not entirely sure that this is
    # a worthwhile tradeoff. Maybe something user-configurable within the
    # environment?
    unescaped_vars = _ParamLookup[object](
        template_instance=template_instance,
        error_collector=error_collector,
        placeholder_on_error='',
        overrides=parent_params,
        valid_param_names=template_xable._templatey_signature.var_names)

    unverified_content = _ParamLookup[object](
        template_instance=template_instance,
        error_collector=error_collector,
        placeholder_on_error='',
        valid_param_names=template_xable._templatey_signature.content_names)

    # First of all: request the loaded and parsed template resource from
    # the caller, all the way back up the recursion chain.
    parsed_template_resource = (yield template_instance)
    # This means there was an error in template loading (or a bug in our code.)
    # Log an info message just in case.
    if parsed_template_resource is None:
        logger.info(
            'Got None in response to template load request. This should '
            + 'always correspond to an error loading the template, which '
            + 'should appear in the final ExceptionGroup. If not, please '
            + 'report a bug with templatey!')
        return
    elif not isinstance(parsed_template_resource, ParsedTemplateResource):
        raise TypeError(
            'Impossible branch: requested template, got something else!',
            parsed_template_resource)

    for template_part in parsed_template_resource.parts:
        yield from _coerce_interpolation(
            template_part,
            template_config,
            unescaped_vars,
            unverified_content,
            all_slots,
            error_collector=error_collector)


@singledispatch
def _coerce_interpolation(
        template_part,
        template_config: TemplateConfig,
        unescaped_vars: dict[str, object],
        unverified_content: dict[str, object],
        all_slots: dict[str, Sequence[TemplateParamsInstance]],
        *,
        error_collector: list[Exception]
        ) -> Generator[
            TemplateParamsInstance | str | FuncExecutionRequest,
            ParsedTemplateResource | None | FuncExecutionResult,
            None]:
    """This is the backstop for interpolation coercion to a string,
    which will simply raise. That error will in turn be caught by either
    the parent template (if nested), or the render driver (if toplevel).
    """
    raise TypeError(
        'Impossible branch: invalid template_part type!',
        template_part)


@_coerce_interpolation.register
def _(
        template_part: str,
        template_config: TemplateConfig,
        unescaped_vars: dict[str, object],
        unverified_content: dict[str, object],
        all_slots: dict[str, Sequence[TemplateParamsInstance]],
        *,
        error_collector: list[Exception]
        ) -> Generator[
            TemplateParamsInstance | str | FuncExecutionRequest,
            ParsedTemplateResource | None | FuncExecutionResult,
            None]:
    """When we encounter a template_part that is an explicit string,
    that means it was literal text within the template itself, which we
    assume to be trusted. Therefore we don't need to do any escaping or
    verification, and can simply yield it back up the call chain.
    """
    yield template_part


@_coerce_interpolation.register
def _(
        template_part: InterpolatedVariable,
        template_config: TemplateConfig,
        unescaped_vars: dict[str, object],
        unverified_content: dict[str, object],
        all_slots: dict[str, Sequence[TemplateParamsInstance]],
        *,
        error_collector: list[Exception]
        ) -> Generator[
            TemplateParamsInstance | str | FuncExecutionRequest,
            ParsedTemplateResource | None | FuncExecutionResult,
            None]:
    """When we encounter an interpolated variable, we need to first
    apply whatever formatting was defined within the template, and then
    sanitize the resulting value as per the variable escaper in the
    template config. Note that the order there is important as a defense
    against obfuscation as a way to bypass the escaper.
    """
    unescaped_val = _apply_format(
        # Note that we've already checked this is defined in
        # _flatten_and_interpolate, so we don't need .get()
        unescaped_vars[template_part.name],
        template_part.conversion,
        template_part.format_spec)
    yield template_config.variable_escaper(unescaped_val)


@_coerce_interpolation.register
def _(
        template_part: InterpolatedContent,
        template_config: TemplateConfig,
        unescaped_vars: dict[str, object],
        unverified_content: dict[str, object],
        all_slots: dict[str, Sequence[TemplateParamsInstance]],
        *,
        error_collector: list[Exception]
        ) -> Generator[
            TemplateParamsInstance | str | FuncExecutionRequest,
            ParsedTemplateResource | None | FuncExecutionResult,
            None]:
    """Interpolated content can take either a simple or complicated
    form. The simple form is just explicit content: some string that was
    passed in as part of the template parameters. The ComplexContent
    form, however, can be used to adjust the content based on the
    variables passed to the template, which can be very useful in
    situations like localization -- for example, getting quantity/plural
    alignment.

    In either case, once we have the final content string to include,
    we need to run a content verifier on it to make sure that it isn't
    breaking anything, as defined within the template config.
    """
    # Note that we've already checked this is defined in
    # _flatten_and_interpolate, so we don't need .get()
    val_from_params = unverified_content[template_part.name]

    if isinstance(val_from_params, str):
        template_config.content_verifier(val_from_params)
        yield val_from_params

    elif isinstance(val_from_params, ComplexContent):
        for content_segment in val_from_params.flatten(unescaped_vars):
            if isinstance(content_segment, str):
                template_config.content_verifier(content_segment)
                yield content_segment

            elif isinstance(content_segment, InterpolatedVariable):
                yield from _coerce_interpolation(
                    content_segment,
                    template_config,
                    unescaped_vars,
                    unverified_content,
                    all_slots,
                    error_collector=error_collector)

            else:
                error_collector.append(_capture_traceback(TypeError(
                    'ComplexContent.flatten() must always return strings '
                    + 'or InterpolatedVariable instances!',
                    content_segment)))

    else:
        error_collector.append(_capture_traceback(TypeError(
            'Interpolated content values must always be strings or '
            + 'ComplexContent instances!', val_from_params)))


@_coerce_interpolation.register
def _(
        template_part: InterpolatedSlot,
        template_config: TemplateConfig,
        unescaped_vars: dict[str, object],
        unverified_content: dict[str, object],
        all_slots: dict[str, Sequence[TemplateParamsInstance]],
        *,
        error_collector: list[Exception]
        ) -> Generator[
            TemplateParamsInstance | str | FuncExecutionRequest,
            ParsedTemplateResource | None | FuncExecutionResult,
            None]:
    """Interpolated slots need to be recursed into -- that's the whole
    point of flattening things out.
    """
    slots_to_recurse = all_slots[template_part.name]
    slot_params_from_parent_template = template_part.params

    for slot_to_recurse in slots_to_recurse:
        try:
            yield from _flatten_and_interpolate(
                slot_to_recurse,
                parent_params=slot_params_from_parent_template,
                error_collector=error_collector)
        # Exceptions can bubble out if eg a nested template wasn't found, or
        # if the signature mismatched, etc.
        except Exception as exc:
            error_collector.append(exc)


@_coerce_interpolation.register
def _(
        template_part: InterpolatedFunctionCall,
        template_config: TemplateConfig,
        unescaped_vars: dict[str, object],
        unverified_content: dict[str, object],
        all_slots: dict[str, Sequence[TemplateParamsInstance]],
        *,
        error_collector: list[Exception]
        ) -> Generator[
            TemplateParamsInstance | str | FuncExecutionRequest,
            ParsedTemplateResource | None | FuncExecutionResult,
            None]:
    """Interpolated function calls are a special case. They need to be
    handled by the outer driver: first, because they might be async,
    second, because it makes more sense to be executing global template
    functions in a global context, and third, because the functions are
    defined all the way up in the template environment, and we don't
    want to need to pass all of the template functions recursively all
    the way down the call stack.
    """
    args = _recursively_coerce_func_execution_params(
        template_part.call_args,
        unescaped_vars=unescaped_vars,
        unverified_content=unverified_content)
    kwargs = _recursively_coerce_func_execution_params(
        template_part.call_kwargs,
        unescaped_vars=unescaped_vars,
        unverified_content=unverified_content)

    function_result = (yield FuncExecutionRequest(
        template_part.name, args=args, kwargs=kwargs))
    if not isinstance(function_result, FuncExecutionResult):
        raise TypeError(
            'Impossible branch: needed template func result, got something '
            + 'else!', function_result)

    if function_result.exc is not None:
        error_collector.append(_capture_traceback(
            TemplateFunctionFailure(
                'Template function raised!', template_part.name),
            from_exc=function_result.exc))
        return

    if function_result.retval is None:
        raise TypeError(
            'Impossible branch: template function retval has None for both '
            + 'exc and retval!')

    for returned_part in function_result.retval:
        if isinstance(returned_part, str):
            yield template_config.variable_escaper(returned_part)

        elif isinstance(returned_part, InjectedValue):
            yield _coerce_injected_value(
                returned_part,
                template_config)

        # Note that interpolated function calls MUST supply the full
        # slot context; you can't add additional variables from the parent
        # template after the function call returns the bound slot!
        elif is_template_instance(returned_part):
            nested_template = cast(TemplateParamsInstance, returned_part)
            yield from _flatten_and_interpolate(
                nested_template,
                error_collector=error_collector)

        else:
            raise TypeError(
                'Invalid return from template function', returned_part)


@overload
def _recursively_coerce_func_execution_params(
        param_value: str,
        *,
        unescaped_vars: _ParamLookup[object],
        unverified_content: _ParamLookup[object]
        ) -> str: ...
@overload
def _recursively_coerce_func_execution_params[K: object, V: object](
        param_value: Mapping[K, V],
        *,
        unescaped_vars: _ParamLookup[object],
        unverified_content: _ParamLookup[object]
        ) -> dict[K, V]: ...
@overload
def _recursively_coerce_func_execution_params[T: object](
        param_value: list[T] | tuple[T],
        *,
        unescaped_vars: _ParamLookup[object],
        unverified_content: _ParamLookup[object]
        ) -> tuple[T]: ...
@overload
def _recursively_coerce_func_execution_params(
        param_value: NestedContentReference | NestedVariableReference,
        *,
        unescaped_vars: _ParamLookup[object],
        unverified_content: _ParamLookup[object]
        ) -> object: ...
@overload
def _recursively_coerce_func_execution_params[T: object](
        param_value: T,
        *,
        unescaped_vars: _ParamLookup[object],
        unverified_content: _ParamLookup[object]
        ) -> T: ...
@singledispatch
def _recursively_coerce_func_execution_params(
        # Note: singledispatch doesn't support type vars
        param_value: object,
        *,
        unescaped_vars: _ParamLookup[object],
        unverified_content: _ParamLookup[object]
        ) -> object:
    """Templatey templates support references to both content and
    variables as call args/kwargs for template functions. They also
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
        unescaped_vars: _ParamLookup[object],
        unverified_content: _ParamLookup[object]
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
        unescaped_vars: _ParamLookup[object],
        unverified_content: _ParamLookup[object]
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
        unescaped_vars: _ParamLookup[object],
        unverified_content: _ParamLookup[object]
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
        unescaped_vars: _ParamLookup[object],
        unverified_content: _ParamLookup[object]
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
    template functions to indicate that the result should be sourced
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
