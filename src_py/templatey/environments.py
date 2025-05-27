import inspect
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal
from typing import Optional
from typing import Protocol
from typing import cast
from typing import runtime_checkable

from templatey._bootstrapping import PARSED_EMPTY_TEMPLATE
from templatey._bootstrapping import EmptyTemplate
from templatey.exceptions import MismatchedRenderColor
from templatey.exceptions import MismatchedTemplateEnvironment
from templatey.exceptions import MismatchedTemplateSignature
from templatey.exceptions import TemplateyException
from templatey.parser import ParsedTemplateResource
from templatey.parser import parse
from templatey.renderer import FuncExecutionRequest
from templatey.renderer import FuncExecutionResult
from templatey.renderer import render_driver
from templatey.templates import InjectedValue
from templatey.templates import TemplateIntersectable
from templatey.templates import TemplateParamsInstance

# Note: strings here will be escaped. InjectedValues may decide whether or not
# escaping should be applied. Nested templates will not be escaped.
EnvFunction = Callable[
    ..., Sequence[str | TemplateParamsInstance | InjectedValue]]
EnvFunctionAsync = Callable[
    ..., Awaitable[Sequence[str | TemplateParamsInstance | InjectedValue]]]


@dataclass(frozen=True, slots=True)
class _TemplateFunctionContainer[F: EnvFunction | EnvFunctionAsync]:
    name: str
    function: F
    signature: inspect.Signature
    is_async: bool


@runtime_checkable
class SyncTemplateLoader[L: object](Protocol):

    def load_sync(
            self,
            template: type[TemplateParamsInstance],
            template_resource_locator: L
            ) -> str:
        """This is responsible for loading the actual template text,
        based on the passed resource locator.
        """
        ...


@runtime_checkable
class AsyncTemplateLoader[L: object](Protocol):

    async def load_async(
            self,
            template: type[TemplateParamsInstance],
            template_resource_locator: L
            ) -> str:
        """This is responsible for loading the actual template text,
        based on the passed resource locator.
        """
        ...


class RenderEnvironment:
    _parsed_template_cache: dict[
        type[TemplateParamsInstance], ParsedTemplateResource]
    _env_functions: dict[str, _TemplateFunctionContainer]
    # We use this to prevent registering template functions after any calls
    # to load() have been made, because it would result in different template
    # functions per template
    _has_loaded_any_template: bool
    _template_loader: SyncTemplateLoader | AsyncTemplateLoader
    strict_interpolation_validation: bool

    def __init__(
            self,
            template_loader: SyncTemplateLoader | AsyncTemplateLoader,
            env_functions:
                Optional[Iterable[EnvFunction | EnvFunctionAsync]] = None,
            # If True, this will make sure that the template interface exactly
            # matches the template text. If False, this will just make sure
            # that the template interface is at least sufficient for the
            # template text.
            strict_interpolation_validation: bool = True):
        self.strict_interpolation_validation = strict_interpolation_validation
        self._has_loaded_any_template = False

        self._env_functions = {}
        if env_functions is not None:
            for function in env_functions:
                self.register_env_function(function)

        self._can_load_sync = isinstance(template_loader, SyncTemplateLoader)
        self._can_load_async = isinstance(template_loader, AsyncTemplateLoader)
        self._template_loader = template_loader
        self._parsed_template_cache = {}

    def register_env_function(
            self,
            env_function: EnvFunction | EnvFunctionAsync,
            *,
            force_async: bool = False,
            with_name: str | None = None):
        """Manually register an environment function with the render
        environment, instead of passing it in to the environment
        constructor.

        This can be used to force a function to be registerd as async,
        in case it was not inferred as such, by passing
        ``force_async=True``.

        Normally, registered functions are assigned their __name__ as
        the function name; manual registration can also be used to
        override this behavior via the ``with_name`` parameter.
        """
        if self._has_loaded_any_template:
            raise TemplateyException(
                'To prevent having different template functions per template, '
                + 'you cannot register new template functions after loading '
                + 'any templates in an environment.')

        if with_name is None:
            function_name = env_function.__name__
        else:
            function_name = with_name

        self._env_functions[function_name] = _TemplateFunctionContainer(
            name=function_name,
            function=env_function,
            signature=inspect.signature(env_function),
            is_async=force_async or _infer_asyncness(env_function))

    async def load_async(
            self,
            template: type[TemplateParamsInstance],
            *,
            override_validation_strictness: None | bool = None,
            force_reload: bool = False
            ) -> ParsedTemplateResource:
        """Loads a template resource from a TemplateParamsInstance
        class. Caches it within the environment and returns the
        resulting ParsedTemplateResource.

        If force_reload is True, bypasses the cache.
        """
        if not self._can_load_async:
            raise TypeError(
                'Current template loader does not support async loading')

        if template is EmptyTemplate:
            return PARSED_EMPTY_TEMPLATE

        if (
            not force_reload
            and template in self._parsed_template_cache
        ):
            return self._parsed_template_cache[template]

        template_loader = cast(AsyncTemplateLoader, self._template_loader)
        template_class = cast(type[TemplateIntersectable], template)
        template_text = await template_loader.load_async(
            template,
            template_class._templatey_resource_locator)
        return self._parse_and_cache(
            template_class,
            template_text,
            override_validation_strictness)

    def load_sync(
            self,
            template: type[TemplateParamsInstance],
            *,
            override_validation_strictness: None | bool = None,
            force_reload: bool = False
            ) -> ParsedTemplateResource:
        """Loads a template resource from a TemplateParamsInstance
        class. Caches it within the environment and returns the
        resulting ParsedTemplateResource.

        If force_reload is True, bypasses the cache.
        """
        if not self._can_load_sync:
            raise TypeError(
                'Current template loader does not support sync loading')

        if template is EmptyTemplate:
            return PARSED_EMPTY_TEMPLATE

        # Note that cache operations here aren't threadsafe, but as long as
        # the underlying resource doesn't change, the worst case is that we
        # load and parse the same template resource multiple times without
        # calling force_reload. This is probably better than wrapping the
        # whole thing in a lock.
        if (
            not force_reload
            and template in self._parsed_template_cache
        ):
            return self._parsed_template_cache[template]

        # The cast here is because it could be a sync and/or async loader,
        # and the type system doesn't know we just verified that via
        # _can_load_sync.
        template_loader = cast(SyncTemplateLoader, self._template_loader)
        template_class = cast(type[TemplateIntersectable], template)
        template_text = template_loader.load_sync(
            template,
            template_class._templatey_resource_locator)
        return self._parse_and_cache(
            template_class,
            template_text,
            override_validation_strictness)

    def _parse_and_cache(
            self,
            template_class: type[TemplateIntersectable],
            template_text: str,
            override_validation_strictness: None | bool
            ) -> ParsedTemplateResource:
        # Note: doing this first for thread safety in the sync case
        self._has_loaded_any_template = True
        try:
            template_config = template_class._templatey_config
            parsed_template_resource = parse(
                template_text, template_config.interpolator)

            if override_validation_strictness is None:
                strict_mode = self.strict_interpolation_validation
            else:
                strict_mode = override_validation_strictness

            self._validate_env_functions(
                template_class, parsed_template_resource,)
            self._validate_template_signature(
                template_class, parsed_template_resource,
                strict_mode=strict_mode)

        except Exception:
            self._has_loaded_any_template = False
            raise

        template = cast(type[TemplateParamsInstance], template_class)
        self._parsed_template_cache[template] = parsed_template_resource
        return parsed_template_resource

    def _validate_env_functions(
            self,
            template_class: type[TemplateIntersectable],
            parsed_template_resource: ParsedTemplateResource
            ) -> Literal[True]:
        """Makes sure that the template environment contains all of the
        template functions referenced in the template text. Returns
        True or raises MismatchedTemplateEnvironment.

        Note that we never use strict mode here, because it would be
        silly to require every single template to call every single
        template function. That's simply not what they're meant to be
        used for!
        """
        # Interestingly, .difference() works here, but plain ``-`` doesn't
        function_mismatch = parsed_template_resource.function_names.difference(
            self._env_functions)
        if function_mismatch:
            raise MismatchedTemplateEnvironment(
                'Template environment functions did not match the template '
                + 'text!', template_class, function_mismatch)

        for (
            function_name, function_calls
        ) in parsed_template_resource.function_calls.items():
            function_container = self._env_functions[function_name]
            for function_call in function_calls:
                try:
                    function_container.signature.bind(
                        *function_call.call_args,
                        **function_call.call_kwargs)
                except TypeError as exc:
                    raise MismatchedTemplateEnvironment(
                        'Template environment function had invalid call '
                        + 'signature', template_class, function_call
                    ) from exc

        return True

    def _validate_template_signature(
            self,
            template_class: type[TemplateIntersectable],
            parsed_template_resource: ParsedTemplateResource,
            *,
            strict_mode: bool
            ) -> Literal[True]:
        """Makes sure that the template signature includes all of the
        names referenced in the template text. Returns True or
        raises MismatchedTemplateSignature.
        """
        template_signature = template_class._templatey_signature
        variable_names = set(template_signature.vars_)
        slot_names = set(template_signature.slots)
        content_names = set(template_signature.content)

        if strict_mode:
            variables_mismatch = (
                parsed_template_resource.variable_names ^ variable_names)
            slot_mismatch = (
                parsed_template_resource.slot_names ^ slot_names)
            content_mismatch = (
                parsed_template_resource.content_names ^ content_names)

        else:
            variables_mismatch = (
                parsed_template_resource.variable_names - variable_names)
            slot_mismatch = (
                parsed_template_resource.slot_names - slot_names)
            content_mismatch = (
                parsed_template_resource.content_names - content_names)

        if variables_mismatch or slot_mismatch or content_mismatch:
            raise MismatchedTemplateSignature(
                'Template interface variables, content, or slots did not '
                + 'match the template text!', template_class,
                variables_mismatch, slot_mismatch, content_mismatch)

        return True

    def render_sync(
            self,
            template_instance: TemplateParamsInstance
            ) -> str:
        output = []
        error_collector = []
        for env_request in render_driver(
            template_instance, output, error_collector
        ):
            for to_load in env_request.to_load:
                env_request.results_loaded[to_load] = self.load_sync(to_load)

            for to_execute in env_request.to_execute:
                env_request.results_executed[to_execute.result_key] = (
                    self.execute_env_function_sync(to_execute))

        if error_collector:
            raise ExceptionGroup('Failed to render template', error_collector)

        return ''.join(output)

    def execute_env_function_sync(
            self,
            request: FuncExecutionRequest
            ) -> FuncExecutionResult:
        try:
            container = self._env_functions[request.name]
            if container.is_async:
                raise MismatchedRenderColor(
                    'Async env funcs cannot be used within render_sync!')

            return FuncExecutionResult(
                name=request.name,
                retval=container.function(*request.args, **request.kwargs),
                exc=None)
        except Exception as exc:
            return FuncExecutionResult(
                name=request.name,
                retval=None,
                exc=exc)

    async def render_async(
            self,
            template_instance: TemplateParamsInstance
            ) -> str:
        output = []
        error_collector = []
        for env_request in render_driver(
            template_instance, output, error_collector
        ):
            for to_load in env_request.to_load:
                env_request.results_loaded[to_load] = await self.load_async(
                    to_load)

            for to_execute in env_request.to_execute:
                env_request.results_executed[to_execute.result_key] = (
                    await self.execute_env_function_async(to_execute))

        if error_collector:
            raise ExceptionGroup('Failed to render template', error_collector)

        return ''.join(output)

    async def execute_env_function_async(
            self,
            request: FuncExecutionRequest
            ) -> FuncExecutionResult:
        try:
            container = self._env_functions[request.name]
            if container.is_async:
                retval = await container.function(
                    *request.args, **request.kwargs)
            else:
                retval = container.function(*request.args, **request.kwargs)

            return FuncExecutionResult(
                name=request.name,
                retval=retval,
                exc=None)
        except Exception as exc:
            return FuncExecutionResult(
                name=request.name,
                retval=None,
                exc=exc)


def _infer_asyncness(env_function: EnvFunction | EnvFunctionAsync) -> bool:
    """Infers, based on the type of the function, whether it should be
    considered sync or async.
    """
    return (
        inspect.iscoroutinefunction(env_function)
        or inspect.isawaitable(env_function))
