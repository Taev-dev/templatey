from __future__ import annotations

import itertools
import logging
from collections.abc import Callable
from collections.abc import Collection
from collections.abc import Hashable
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import KW_ONLY
from dataclasses import dataclass
from dataclasses import field
from functools import partial
from functools import singledispatch
from typing import NamedTuple
from typing import cast
from typing import overload

from templatey._annotations import InterfaceAnnotationFlavor
from templatey._bootstrapping import EMPTY_TEMPLATE_INSTANCE
from templatey._bootstrapping import EMPTY_TEMPLATE_XABLE
from templatey._provenance import Provenance
from templatey._provenance import ProvenanceNode
from templatey._signature import TemplateSignature
from templatey._slot_tree import SlotTreeNode
from templatey._slot_tree import extract_dynamic_template_classes
from templatey._types import TemplateClass
from templatey._types import TemplateIntersectable
from templatey._types import TemplateParamsInstance
from templatey._types import is_template_instance
from templatey.exceptions import MismatchedTemplateSignature
from templatey.exceptions import TemplateFunctionFailure
from templatey.parser import InterpolatedContent
from templatey.parser import InterpolatedFunctionCall
from templatey.parser import InterpolatedSlot
from templatey.parser import InterpolatedVariable
from templatey.parser import InterpolationConfig
from templatey.parser import NestedContentReference
from templatey.parser import NestedVariableReference
from templatey.parser import ParsedTemplateResource
from templatey.templates import ComplexContent
from templatey.templates import InjectedValue
from templatey.templates import TemplateConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FuncExecutionRequest:
    name: str
    args: Iterable[object]
    kwargs: Mapping[str, object]
    result_key: _PrecallCacheKey
    provenance: Provenance


@dataclass(frozen=True, slots=True)
class FuncExecutionResult:
    # Note: must match signature from TemplateFunction!
    name: str
    retval: Sequence[str | TemplateParamsInstance | InjectedValue] | None
    exc: Exception | None

    def filter_injectables(self) -> Iterable[TemplateParamsInstance]:
        if self.retval is not None:
            for item in self.retval:
                if is_template_instance(item):
                    # This doesn't fully work; because of the missing
                    # intersection type, there's nothing linking the xable that
                    # this checks for with the params instance that the type
                    # expects us to yield back
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


# Yes, this is a larger function than it should be.
# But function calls in python are slow, and we actually kinda care about
# performance here.
# TODO: look into solutions that would allow for inlining functions, so you
# could carve this up into separate functions. Can this be done without import
# hooks?
def render_driver(  # noqa: C901, PLR0912, PLR0915
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
    template_xable = cast(TemplateIntersectable, template_instance)
    root_template_preload = context.template_preload[type(template_instance)]
    render_stack: list[_RenderStackFrame] = [
        _RenderStackFrame(
            parts=root_template_preload.parts,
            part_count=root_template_preload.part_count,
            config=template_xable._templatey_config,
            signature=template_xable._templatey_signature,
            provenance=Provenance((
                ProvenanceNode(
                    encloser_slot_key='',
                    encloser_slot_index=-1,
                    instance_id=id(template_instance),
                    instance=template_instance),)),
            instance=template_instance,
            prerenderers=template_xable._templatey_prerenderers)]

    while render_stack:
        # Quick note on the following code: you might think these isinstance
        # calls are slow, and that you could speed things up by, say, using
        # some kind of sentinel value in a slot. This is incorrect! (At least
        # with 3.13). Isinstance is actually the fastest (and clearest) way
        # to do it.
        try:
            render_frame = render_stack[-1]
            if render_frame.exhausted:
                render_stack.pop()
                continue

            next_part = render_frame.parts[render_frame.part_index]

            # Strings are hardest to deal with because they're containers, so
            # just get that out of the way first
            if isinstance(next_part, str):
                render_frame.part_index += 1
                output.append(next_part)

            elif isinstance(next_part, InterpolatedVariable):
                render_frame.part_index += 1
                unescaped_vars = _ParamLookup(
                    provenance=render_frame.provenance,
                    template_preload=context.template_preload,
                    param_flavor=InterfaceAnnotationFlavor.VARIABLE,
                    error_collector=error_collector,
                    placeholder_on_error='')

                raw_val = unescaped_vars[next_part.name]
                prerenderer = getattr(
                    render_frame.prerenderers, next_part.name, None)
                if prerenderer is None:
                    prerender = raw_val
                else:
                    prerender = cast(str | None, prerenderer(raw_val))

                if prerender is not None:
                    unescaped_val = _apply_format(
                        prerender,
                        next_part.config)
                    escaped_val = render_frame.config.variable_escaper(
                        unescaped_val)
                    # Note that variable interpolations don't support affixes!
                    output.append(escaped_val)

            elif isinstance(next_part, InterpolatedContent):
                render_frame.part_index += 1
                unverified_content = _ParamLookup(
                    provenance=render_frame.provenance,
                    template_preload=context.template_preload,
                    param_flavor=InterfaceAnnotationFlavor.CONTENT,
                    error_collector=error_collector,
                    placeholder_on_error='')
                val_from_params = unverified_content[next_part.name]

                if isinstance(val_from_params, ComplexContent):
                    output.extend(_render_complex_content(
                        val_from_params,
                        _ParamLookup(
                            provenance=render_frame.provenance,
                            template_preload=context.template_preload,
                            param_flavor=InterfaceAnnotationFlavor.VARIABLE,
                            error_collector=error_collector,
                            placeholder_on_error=''),
                        render_frame.config,
                        next_part.config,
                        render_frame.prerenderers,
                        error_collector))

                else:
                    prerenderer = getattr(
                        render_frame.prerenderers, next_part.name, None)
                    if prerenderer is None:
                        prerender = val_from_params
                    else:
                        prerender = cast(
                            str | None, prerenderer(val_from_params))

                    # As usual, values of None are omitted
                    if prerender is not None:
                        formatted_val = _apply_format(
                            prerender,
                            next_part.config)
                        render_frame.config.content_verifier(formatted_val)
                        output.extend(
                            next_part.config.apply_affix(formatted_val))

            # Slots get a little bit more complicated, but the general idea is
            # to append a stack frame for each depth level. We maintain the
            # state of which instance we're on within the stack frame.
            elif isinstance(next_part, InterpolatedSlot):
                slot_instance_index = render_frame.slot_instance_index
                if slot_instance_index == 0:
                    slot_instances = getattr(
                        render_frame.instance, next_part.name)
                    slot_instance_count = len(slot_instances)

                    if slot_instance_count > 0:
                        render_frame.slot_instance_count = slot_instance_count
                        render_frame.slot_instances = slot_instances
                    # Note that this is also important to skip prefix/suffix.
                    else:
                        render_frame.part_index += 1
                        continue

                else:
                    # Note that we skip this entirely if the slot instance
                    # count is zero, so by definition, if we hit this branch,
                    # we need a suffix.
                    output.extend(next_part.config.apply_suffix_iter())
                    slot_instance_count = render_frame.slot_instance_count
                    # We've exhausted the instances; reset the state for the
                    # next slot.
                    if render_frame.slot_instance_index >= slot_instance_count:
                        render_frame.slot_instance_index = 0
                        render_frame.slot_instance_count = 0
                        # Note: we're deliberately skipping the slot instances,
                        # because it'll just get overwritten the next time
                        # around, so this is faster (though it doesn't free
                        # memory as quickly)
                        render_frame.part_index += 1
                        continue

                    slot_instances = render_frame.slot_instances

                # Remember: we skip this entirely if the slot instance count
                # is zero.
                output.extend(next_part.config.apply_prefix_iter())
                slot_instance = slot_instances[slot_instance_index]
                # Note that this needs to support both union slot
                # types, and (eventually) dynamic slot types, hence
                # doing this on every iteration instead of
                # precalculating it for the whole interpolated slot
                slot_instance_preload = context.template_preload[
                    type(slot_instance)]
                slot_instance_xable = cast(
                    TemplateIntersectable, slot_instance)
                render_stack.append(
                    _RenderStackFrame(
                        instance=slot_instance,
                        parts=slot_instance_preload.parts,
                        part_count=slot_instance_preload.part_count,
                        config=slot_instance_xable._templatey_config,
                        signature=slot_instance_xable._templatey_signature,
                        provenance=render_frame.provenance.with_appended(
                            ProvenanceNode(
                                encloser_slot_key=next_part.name,
                                encloser_slot_index=slot_instance_index,
                                instance_id=id(slot_instance),
                                instance=slot_instance)),
                        prerenderers=
                            slot_instance_xable._templatey_prerenderers))

                render_frame.slot_instance_index += 1

            elif isinstance(next_part, InterpolatedFunctionCall):
                render_frame.part_index += 1
                execution_result = context.function_precall[
                    _get_precall_cache_key(render_frame.provenance, next_part)]
                nested_render_node = _build_render_frame_for_func_result(
                    render_frame.instance,
                    render_frame.provenance,
                    next_part,
                    execution_result,
                    render_frame.config,
                    error_collector)
                if nested_render_node is not None:
                    render_stack.append(nested_render_node)

            # Similar to slots, but different enough that it warrants a
            # separate approach. Trust me, I tried to unify them, and 1. I
            # never got it fully working, 2. it was a pretty big hack (a
            # virtual slot mechanism with some __getattr__ shenanigans), and
            # 3. it created way more problems than it was worth.
            # More info in the docstring for _InjectedInstanceContainer.
            elif isinstance(next_part, _InjectedInstanceContainer):
                render_frame.part_index += 1
                injected_instance = next_part.instance
                # Note that this needs to support both union slot
                # types, and (eventually) dynamic slot types, hence
                # doing this on every iteration instead of
                # precalculating it for the whole interpolated slot
                injected_instance_preload = context.template_preload[
                    type(injected_instance)]
                injected_instance_xable = cast(
                    TemplateIntersectable, injected_instance)
                render_stack.append(
                    _RenderStackFrame(
                        instance=injected_instance,
                        parts=injected_instance_preload.parts,
                        part_count=injected_instance_preload.part_count,
                        config=injected_instance_xable._templatey_config,
                        signature=injected_instance_xable._templatey_signature,
                        # Note that the correct ``from_injection`` value is
                        # added when creating the current stack frame.
                        provenance=render_frame.provenance.with_appended(
                            ProvenanceNode(
                                encloser_slot_key='',
                                encloser_slot_index=-1,
                                instance_id=id(injected_instance),
                                instance=injected_instance),),
                        prerenderers=
                            injected_instance_xable._templatey_prerenderers))

            else:
                raise TypeError(
                    'Templatey internal error: unknown template part!',
                    next_part)

        # Note: this could be eg a lookup error because of a missing variable.
        # This isn't redundant with the error collection within prepopulation.
        except Exception as exc:
            error_collector.append(exc)


@dataclass(slots=True)
class _InjectedInstanceContainer:
    """When env functions inject instances of templates, the render
    stack gets into a bit of a weird state. We want to avoid needing to
    extend the stack by one for each and every injected template and/or
    stringified value that the function returned. But we can't modify
    the existing parts of a frame; first, it's a sometimes (always?) a
    tuple, but more importantly, it would screw up our entire indexing
    mechanism.

    In theory we could add a "pending injections" sub-stack or something
    to the _RenderStackFrame, but then we'd be stuck checking for that
    during every single iteration of the render driver loop.

    So instead, we do this: wrap the instance into a container, and
    then (as the LAST, and therefore SLOWEST, check in the render driver
    loop) we can check for the container, and have special logic for
    handling them there.
    """
    instance: TemplateParamsInstance


@dataclass(slots=True)
class _RenderStackFrame:
    instance: TemplateParamsInstance
    config: TemplateConfig
    signature: TemplateSignature
    provenance: Provenance
    prerenderers: NamedTuple
    parts: Sequence[
        str
        | InterpolatedSlot
        | InterpolatedContent
        | InterpolatedVariable
        | InterpolatedFunctionCall
        | _InjectedInstanceContainer]

    _: KW_ONLY
    part_count: int
    part_index: int = field(default=0, init=False)
    slot_instance_count: int = field(default=0, init=False)
    slot_instance_index: int = field(default=0, init=False)
    slot_instances: Sequence[TemplateParamsInstance] = field(init=False)

    @property
    def exhausted(self) -> bool:
        # Note that the slot index exhaustion is handled separately, by
        # controlling exactly when the part index is incremented
        return self.part_index >= self.part_count


@dataclass(slots=True)
class _PrepopulationBatch:
    injection_provenance: Provenance | None
    # Note that this might be an injected template!
    local_root_instance: TemplateParamsInstance
    function_backlog: list[FuncExecutionRequest]
    template_backlog: set[TemplateClass]

    def extract_injections(
            self,
            function_precall:
                dict[_PrecallCacheKey, FuncExecutionResult]
            ) -> list[_PrepopulationBatch]:
        result = []
        for exe_req in self.function_backlog:
            result_cache_key = exe_req.result_key
            function_result = function_precall[result_cache_key]

            for injected_template in function_result.filter_injectables():
                injected_xable = cast(
                    TemplateIntersectable, injected_template)

                result.append(_PrepopulationBatch(
                    injection_provenance=exe_req.provenance,
                    local_root_instance=injected_template,
                    function_backlog=[],
                    template_backlog=set(
                        injected_xable._templatey_signature
                        .included_template_classes)))

        return result

    def extract_invocations(
            self,
            template_preload:
                dict[TemplateClass, ParsedTemplateResource],
            error_collector: list[Exception],
            ) -> _PrepopulationBatch | None:
        function_backlog = []
        injection_provenance = self.injection_provenance
        root_instance = self.local_root_instance
        root_xable = cast(TemplateIntersectable, root_instance)
        slot_tree_lookup = root_xable._templatey_signature._slot_tree_lookup

        for template_cls in self.template_backlog:
            abstract_calls = (
                template_preload[template_cls].function_calls)

            if abstract_calls:
                # Remember: for the root template, it might or might not
                # exist in the slot tree, depending on whether or not it has
                # recursion loops. That being said, we want to be
                # defensive, and not assume that BECAUSE it's missing, it's
                # automatically the root instance.
                if template_cls is type(root_instance):
                    slot_tree_root = slot_tree_lookup.get(
                        template_cls,
                        SlotTreeNode(is_terminus=True))
                else:
                    slot_tree_root = slot_tree_lookup[template_cls]

                # Oof. The combinatorics here are brutal.
                for provenance, abstract_call in itertools.product(
                    Provenance.from_slot_tree(
                        root_instance,
                        slot_tree_root,
                        from_injection=injection_provenance),
                    itertools.chain.from_iterable(abstract_calls.values())
                ):
                    unescaped_vars = _ParamLookup(
                        provenance=provenance,
                        template_preload=template_preload,
                        param_flavor=InterfaceAnnotationFlavor.VARIABLE,
                        error_collector=error_collector,
                        placeholder_on_error='')
                    unverified_content = _ParamLookup(
                        provenance=provenance,
                        template_preload=template_preload,
                        param_flavor=InterfaceAnnotationFlavor.CONTENT,
                        error_collector=error_collector,
                        placeholder_on_error='')

                    # Note that the full call signature is **defined**
                    # within the parsed template body, but it may
                    # **reference** vars and/or content within the template
                    # instance.
                    args = _recursively_coerce_func_execution_params(
                        abstract_call.call_args,
                        unescaped_vars=unescaped_vars,
                        unverified_content=unverified_content)
                    kwargs = _recursively_coerce_func_execution_params(
                        abstract_call.call_kwargs,
                        unescaped_vars=unescaped_vars,
                        unverified_content=unverified_content)

                    if abstract_call.call_args_exp is not None:
                        args = (*args, *cast(
                            Iterable,
                            _recursively_coerce_func_execution_params(
                                abstract_call.call_args_exp,
                                unescaped_vars=unescaped_vars,
                                unverified_content=unverified_content)))

                    if abstract_call.call_kwargs_exp is not None:
                        kwargs.update(cast(
                            Mapping,
                            _recursively_coerce_func_execution_params(
                                abstract_call.call_kwargs_exp,
                                unescaped_vars=unescaped_vars,
                                unverified_content=unverified_content)))

                    result_cache_key = _get_precall_cache_key(
                        provenance, abstract_call)
                    function_backlog.append(
                        FuncExecutionRequest(
                            abstract_call.name,
                            args=args,
                            kwargs=kwargs,
                            result_key=result_cache_key,
                            provenance=provenance))

        if function_backlog:
            return _PrepopulationBatch(
                injection_provenance=injection_provenance,
                local_root_instance=self.local_root_instance,
                function_backlog=function_backlog,
                template_backlog=set())


@dataclass(slots=True)
class _RenderContext:
    template_preload: dict[TemplateClass, ParsedTemplateResource]
    function_precall: dict[_PrecallCacheKey, FuncExecutionResult]
    error_collector: list[Exception]

    def prepopulate(
            self,
            root_template_instance: TemplateParamsInstance
            ) -> Iterable[RenderEnvRequest]:
        """For the passed root template, populates the template_preload
        and function_precall until either all resources have been
        prepared, or it needs help from the render environment.
        """
        root_template_xable = cast(
            TemplateIntersectable, root_template_instance)
        template_preload: \
            dict[TemplateClass, ParsedTemplateResource] = self.template_preload
        function_precall: \
            dict[_PrecallCacheKey, FuncExecutionResult] = self.function_precall

        batches: list[_PrepopulationBatch] = [_PrepopulationBatch(
            injection_provenance=None,
            local_root_instance=root_template_instance,
            function_backlog=[],
            # Note that this always includes the root template class
            template_backlog=set(
                root_template_xable
                ._templatey_signature.included_template_classes))]

        while batches:
            to_load = set()
            to_execute = []
            # Note that it's important that we're modifying the params for
            # the RenderEnvRequest and not the batches, because the batches
            # will determine where we look for additional injections and/or
            # invocations, regardless of whether or not an execution or load
            # was actually required.
            for batch in batches:
                local_root_instance = batch.local_root_instance
                to_load.update(batch.template_backlog)
                to_execute.extend(batch.function_backlog)
                to_load.update(extract_dynamic_template_classes(
                    local_root_instance,
                    cast(TemplateIntersectable, local_root_instance)
                        ._templatey_signature._dynamic_class_slot_tree))
            to_load.difference_update(template_preload)

            yield RenderEnvRequest(
                to_load=to_load,
                to_execute=to_execute,
                error_collector=self.error_collector,
                results_loaded=template_preload,
                results_executed=function_precall)

            completed_batches = batches
            batches = []
            # Yes, it's a little awkward to re-iterate over this, but we need
            # to keep track of the injection provenance, and this is the most
            # convenient way to do it.
            # Also note: it's crticial that we're still looking at batches
            # and not to_load/to_execute, since those will skip already-loaded
            # templates, which would result in us skipping needed invocations.
            for completed_batch in completed_batches:
                batches.extend(completed_batch.extract_injections(
                    function_precall))
                next_batch = completed_batch.extract_invocations(
                    template_preload,
                    self.error_collector)
                if next_batch is not None:
                    batches.append(next_batch)


type _PrecallExecutionRequest = tuple[
    Provenance, InterpolatedFunctionCall]
type _PrecallCacheKey = Hashable


def _get_precall_cache_key(
        provenance: Provenance,
        interpolated_call: InterpolatedFunctionCall
        ) -> _PrecallCacheKey:
    """For a particular template instance and interpolated function
    call, creates the hashable cache key to be used for the render
    context.
    """
    # Note that template provenance includes the actual current template
    # instance, so by definition, this encodes exactly which function call
    # is being referenced
    return (provenance, interpolated_call)


def _render_complex_content(
        complex_content: ComplexContent,
        unescaped_vars: _ParamLookup,
        template_config: TemplateConfig,
        interpolation_config: InterpolationConfig,
        prerenderers: NamedTuple,
        error_collector: list[Exception],
        ) -> Iterable[str]:
    try:
        extracted_vars = {
            key: unescaped_vars[key] for key in complex_content.dependencies}
        extracted_prerenderers = {
            key: getattr(prerenderers, key, None)
            for key in complex_content.dependencies}

        for content_segment in complex_content.flatten(
            extracted_vars, interpolation_config, extracted_prerenderers
        ):
            if isinstance(content_segment, InjectedValue):
                raw_val = content_segment.value
                if raw_val is None:
                    continue

                unescaped_val = _apply_format(raw_val, content_segment.config)

                if content_segment.use_variable_escaper:
                    escaped_val = template_config.variable_escaper(
                        unescaped_val)
                else:
                    escaped_val = unescaped_val

                if content_segment.use_content_verifier:
                    template_config.content_verifier(escaped_val)

                yield escaped_val

            # Note: as usual, None values get omitted!
            elif content_segment is not None:
                formatted_val = _apply_format(
                    content_segment, interpolation_config)
                template_config.content_verifier(formatted_val)
                yield formatted_val

    except Exception as exc:
        exc.add_note('Failed to render complex content!')
        error_collector.append(exc)


def _build_render_frame_for_func_result(  # noqa: C901
        enclosing_instance: TemplateParamsInstance,
        enclosing_provenance: Provenance,
        abstract_call: InterpolatedFunctionCall,
        execution_result: FuncExecutionResult,
        template_config: TemplateConfig,
        error_collector: list[Exception]
        ) -> _RenderStackFrame | None:
    """This constructs a _RenderNode for the given execution result and
    returns it (or None, if there was an error).
    """
    injected_templates: list[tuple[int, TemplateParamsInstance]] = []
    resulting_parts: list[str | _InjectedInstanceContainer] = []
    if execution_result.exc is None:
        if execution_result.retval is None:
            raise TypeError(
                'Impossible branch! Malformed func exe result',
                execution_result)

        for index, result_part in enumerate(execution_result.retval):
            if isinstance(result_part, str):
                resulting_parts.append(
                    template_config.variable_escaper(result_part))
            elif isinstance(result_part, InjectedValue):
                resulting_parts.append(
                    _coerce_injected_value(result_part, template_config))
            elif is_template_instance(result_part):
                injected_templates.append((index, result_part))
                # This is just a placeholder; it gets overwritten in
                # _build_render_stack_extension
                resulting_parts.append('')
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

    if injected_templates:
        for index, template_instance in injected_templates:
            resulting_parts[index] = _InjectedInstanceContainer(
                template_instance)

        return _RenderStackFrame(
            parts=resulting_parts,
            part_count=len(resulting_parts),
            config=EMPTY_TEMPLATE_XABLE._templatey_config,
            signature=EMPTY_TEMPLATE_XABLE._templatey_signature,
            # Note: keep this empty here, because we need the instance info
            # to match the injected template, and the whole idea here is to
            # avoid a bunch of extraneous stack frames. We'll add in the
            # correct initial node in the render driver code, where we deal
            # with _InjectedInstanceContainers
            provenance=Provenance(from_injection=enclosing_provenance),
            instance=EMPTY_TEMPLATE_INSTANCE,
            prerenderers=EMPTY_TEMPLATE_XABLE._templatey_prerenderers)

    elif resulting_parts:
        return _RenderStackFrame(
            parts=resulting_parts,
            part_count=len(resulting_parts),
            config=EMPTY_TEMPLATE_XABLE._templatey_config,
            signature=EMPTY_TEMPLATE_XABLE._templatey_signature,
            provenance=Provenance(),
            instance=EMPTY_TEMPLATE_INSTANCE,
            prerenderers=EMPTY_TEMPLATE_XABLE._templatey_prerenderers)


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
    provenance: Provenance
    error_collector: list[Exception]
    placeholder_on_error: object
    lookup: Callable[[str], object]
    param_flavor: InterfaceAnnotationFlavor

    def __init__(
            self,
            provenance: Provenance,
            template_preload: dict[TemplateClass, ParsedTemplateResource],
            param_flavor: InterfaceAnnotationFlavor,
            error_collector: list[Exception],
            placeholder_on_error: object):
        self.error_collector = error_collector
        self.placeholder_on_error = placeholder_on_error
        self.provenance = provenance
        self.param_flavor = param_flavor

        if param_flavor is InterfaceAnnotationFlavor.CONTENT:
            self.lookup = partial(
                provenance.bind_content,
                template_preload=template_preload)
        elif param_flavor is InterfaceAnnotationFlavor.VARIABLE:
            self.lookup = partial(
                provenance.bind_variable,
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
                    self.provenance.slotpath[-1].instance,
                    name),
                from_exc=exc))
            return self.placeholder_on_error

    def __len__(self) -> int:
        # Note: this is going to be less commonly used (presumably) than just
        # getitem (the only external access to this is through complex content
        # flattening), so don't precalculate this during __init__
        template_instance = self.provenance.slotpath[-1].instance
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
        template_instance = self.provenance.slotpath[-1].instance
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


def _apply_format(raw_value, config: InterpolationConfig) -> str:
    """For both interpolated variables and injected values, we allow
    format specs and conversions to be supplied. We need to actually
    apply these, but the stdlib doesn't really give us a good way of
    doing that. So this is how we do that instead.
    """
    # hot path go fast
    if config is None or config.fmt is None:
        # Note: yes, strings can be formatted with eg padding, but we literally
        # just checked to make sure that there was no format spec, so format
        # would have nothing to do here!
        if isinstance(raw_value, str):
            formatted_value = raw_value
        else:
            formatted_value = format(raw_value)

    else:
        formatted_value = format(raw_value, config.fmt)

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
    environment functions and complex content to indicate that the
    result should be sourced from the variables and/or the content of
    the current render call. This function is responsible for converting
    the ``InjectedValue`` instance into the final resulting string to
    render.
    """
    unescaped_value = _apply_format(
        injected_value.value,
        injected_value.config)

    if injected_value.use_variable_escaper:
        escapish_value = template_config.variable_escaper(unescaped_value)
    else:
        escapish_value = unescaped_value

    if injected_value.use_content_verifier:
        template_config.content_verifier(escapish_value)

    return escapish_value
