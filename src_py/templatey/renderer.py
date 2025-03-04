from __future__ import annotations

import typing
from collections.abc import Collection
from collections.abc import Generator
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from functools import singledispatch
from typing import cast
from typing import overload

from templatey.exceptions import IncompleteTemplateParams
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
from templatey.templates import TemplateConfig
from templatey.templates import TemplateIntersectable
from templatey.templates import TemplateParamsInstance
from templatey.templates import is_template_instance

if typing.TYPE_CHECKING:
    from templatey.environments import RenderEnvironment


@dataclass
class FuncExecutionRequest:
    name: str
    args: Iterable[object]
    kwargs: Mapping[str, object]


@dataclass
class FuncExecutionResult:
    # Note: must match signature from TemplateFunction!
    retval: Sequence[str | TemplateParamsInstance | InjectedValue] | None
    exc: Exception | None


def render_driver_sync(
        template_instance: TemplateParamsInstance,
        render_environment: RenderEnvironment
        ) -> Iterable[str]:
    """This drives the _flatten_and_interpolate coroutine when
    running in sync mode, and collects all of its errors. More
    details there!
    """
    delegated: TemplateParamsInstance | str | Exception
    requested_help: None | ParsedTemplateResource | FuncExecutionResult = None

    render_recursor = _flatten_and_interpolate(template_instance)
    errors = []
    # Note that we **cannot use a for loop here.** The return value of
    # the call to .send() is the next yield, so if we try to mix together
    # both a ``for flattened_str_or_template_request in ...`` and a
    # ``render_recursor.send(...)``, we skip over flattened strings.
    try:
        while True:
            # Note that it's important that we've pre-initialized the
            # bubbled_request to None, since the first .send() call
            # MUST ALWAYS be None (as per python spec).
            delegated = render_recursor.send(requested_help)
            # Always reset this immediately after sending a value, so that
            # we're less likely to accidentally send the same template
            # twice due to some other bug (defense in depth / fail loudly)
            requested_help = None
            # The flattening resulted in a string value, ready for final
            # rendering
            if isinstance(delegated, str):
                yield delegated

            # The flattening resulted in an error. Collect it so we can
            # raise them all at once at the end.
            # TODO: move exception handling into flatten_and_interpolate.
            # Use the generator return value (which becomes the return value
            # of ``yield from`` to collect them, and then get it from the
            # StopIteration at the end of driving.)
            elif isinstance(delegated, Exception):
                errors.append(delegated)

            elif isinstance(delegated, FuncExecutionRequest):
                requested_help = (
                    render_environment.execute_template_function_sync(
                        delegated))

            # The flattening requires a template resource to continue, so
            # we need to load it before continuing.
            else:
                required_template = type(delegated)
                requested_help = render_environment.load_sync(
                    required_template)

    except StopIteration:
        pass

    # This can happen if there's an error at the outermost template. We want
    # to make sure we always raise an ExceptionGroup for any and all template
    # errors, regardless of depth, so we want to capture it and re-wrap it.
    except Exception as exc:
        errors.append(exc)

    if errors:
        raise ExceptionGroup('Failed to render template', errors)


def _flatten_and_interpolate(
        template_instance: TemplateParamsInstance,
        *,
        parent_params: dict[str, object] | None = None
        ) -> Generator[
            TemplateParamsInstance | str | Exception | FuncExecutionRequest,
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
    all_slots: dict[str, Sequence[TemplateParamsInstance]] = {}
    for slot_name in template_xable._templatey_signature.slots:
        slot_value = getattr(template_instance, slot_name)
        if slot_value is ...:
            yield _capture_traceback(IncompleteTemplateParams(
                    'Missing slot value!', template_instance, slot_name))

        all_slots[slot_name] = slot_value

    # TODO: question: is it worth it to also do type checking of the
    # arguments? This would probably imply something like pydantic if you
    # want to have it be robust, but I'm not entirely sure that this is
    # a worthwhile tradeoff. Maybe something user-configurable within the
    # environment?
    if parent_params is None:
        unescaped_vars: dict[str, object] = {}
    else:
        unescaped_vars: dict[str, object] = parent_params

    for var_name in template_xable._templatey_signature.vars_:
        var_value = getattr(template_instance, var_name)
        if var_value is ...:
            # Note: don't combine with above. This is merging in with the
            # parent params; we want explicit vars passed to overwrite
            # the vars from the template
            if var_name not in unescaped_vars:
                yield _capture_traceback(IncompleteTemplateParams(
                        'Missing var value!', template_instance, var_name))

        else:
            unescaped_vars[var_name] = var_value

    unverified_content = {
        content_name: getattr(template_instance, content_name)
        for content_name in template_xable._templatey_signature.content}

    # First of all: request the loaded and parsed template resource from
    # the caller, all the way back up the recursion chain.
    parsed_template_resource = (yield template_instance)
    if not isinstance(parsed_template_resource, ParsedTemplateResource):
        raise TypeError(
            'Impossible branch: requested template, got something else!',
            parsed_template_resource)

    for template_part in parsed_template_resource.parts:
        yield from _coerce_interpolation(
            template_part,
            template_config,
            unescaped_vars,
            unverified_content,
            all_slots)


@singledispatch
def _coerce_interpolation(
        template_part,
        template_config: TemplateConfig,
        unescaped_vars: dict[str, object],
        unverified_content: dict[str, object],
        all_slots: dict[str, Sequence[TemplateParamsInstance]],
        ) -> Generator[
            TemplateParamsInstance | str | Exception | FuncExecutionRequest,
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
        ) -> Generator[
            TemplateParamsInstance | str | Exception | FuncExecutionRequest,
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
        ) -> Generator[
            TemplateParamsInstance | str | Exception | FuncExecutionRequest,
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
        ) -> Generator[
            TemplateParamsInstance | str | Exception | FuncExecutionRequest,
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
                    all_slots)

            else:
                yield _capture_traceback(TypeError(
                    'ComplexContent.flatten() must always return strings '
                    + 'or InterpolatedVariable instances!',
                    content_segment))

    else:
        yield _capture_traceback(TypeError(
            'Interpolated content values must always be strings or '
            + 'ComplexContent instances!', val_from_params))


@_coerce_interpolation.register
def _(
        template_part: InterpolatedSlot,
        template_config: TemplateConfig,
        unescaped_vars: dict[str, object],
        unverified_content: dict[str, object],
        all_slots: dict[str, Sequence[TemplateParamsInstance]],
        ) -> Generator[
            TemplateParamsInstance | str | Exception | FuncExecutionRequest,
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
                parent_params=slot_params_from_parent_template)
        # Exceptions can bubble out if eg a nested template wasn't found, or
        # if the signature mismatched, etc.
        except Exception as exc:
            yield exc


@_coerce_interpolation.register
def _(
        template_part: InterpolatedFunctionCall,
        template_config: TemplateConfig,
        unescaped_vars: dict[str, object],
        unverified_content: dict[str, object],
        all_slots: dict[str, Sequence[TemplateParamsInstance]],
        ) -> Generator[
            TemplateParamsInstance | str | Exception | FuncExecutionRequest,
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
        yield _capture_traceback(
            TemplateFunctionFailure(
                'Template function raised!', template_part.name),
            from_exc=function_result.exc)
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
            yield from _flatten_and_interpolate(nested_template)

        else:
            raise TypeError(
                'Invalid return from template function', returned_part)


@overload
def _recursively_coerce_func_execution_params(
        param_value: str,
        *,
        unescaped_vars: dict[str, object],
        unverified_content: dict[str, object]
        ) -> str: ...
@overload
def _recursively_coerce_func_execution_params[K: object, V: object](
        param_value: Mapping[K, V],
        *,
        unescaped_vars: dict[str, object],
        unverified_content: dict[str, object]
        ) -> dict[K, V]: ...
@overload
def _recursively_coerce_func_execution_params[T: object](
        param_value: list[T] | tuple[T],
        *,
        unescaped_vars: dict[str, object],
        unverified_content: dict[str, object]
        ) -> tuple[T]: ...
@overload
def _recursively_coerce_func_execution_params(
        param_value: NestedContentReference | NestedVariableReference,
        *,
        unescaped_vars: dict[str, object],
        unverified_content: dict[str, object]
        ) -> object: ...
@overload
def _recursively_coerce_func_execution_params[T: object](
        param_value: T,
        *,
        unescaped_vars: dict[str, object],
        unverified_content: dict[str, object]
        ) -> T: ...
@singledispatch
def _recursively_coerce_func_execution_params(
        # Note: singledispatch doesn't support type vars
        param_value: object,
        *,
        unescaped_vars: dict[str, object],
        unverified_content: dict[str, object]
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
        param_value: list | tuple | Mapping,
        *,
        unescaped_vars: dict[str, object],
        unverified_content: dict[str, object]
        ) -> tuple | dict:
    """Again, in the container case, we want to create a new copy of
    the container, replacing its values with the recursive call.
    Note that the keys in nested dictionaries cannot be references,
    only the values.
    """
    if isinstance(param_value, Mapping):
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
        unescaped_vars: dict[str, object],
        unverified_content: dict[str, object]
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
        unescaped_vars: dict[str, object],
        unverified_content: dict[str, object]
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
        unescaped_vars: dict[str, object],
        unverified_content: dict[str, object]
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
