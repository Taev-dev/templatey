from decimal import Decimal
from typing import cast
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from templatey.interpolators import NamedInterpolator
from templatey.parser import InterpolatedFunctionCall
from templatey.parser import InterpolatedVariable
from templatey.parser import InterpolationConfig
from templatey.parser import LiteralTemplateString
from templatey.parser import NestedContentReference
from templatey.parser import NestedVariableReference
from templatey.parser import ParsedTemplateResource
from templatey.renderer import FuncExecutionResult
from templatey.renderer import _apply_format
from templatey.renderer import _capture_traceback
from templatey.renderer import _coerce_injected_value
from templatey.renderer import _ParamLookup
from templatey.renderer import _recursively_coerce_func_execution_params
from templatey.renderer import _RenderContext
from templatey.renderer import render_driver
from templatey.templates import InjectedValue
from templatey.templates import TemplateConfig
from templatey.templates import Var
from templatey.templates import template

from templatey_testutils import fake_template_config
from templatey_testutils import zderr_template_config


def _noop_escaper(value):
    return value


def _noop_verifier(value):
    return True


@pytest.fixture
def new_fake_template_config():
    """This creates a completely new fake template config, so that the
    return value of the escaper can be configured without affecting
    other tests.
    """
    return TemplateConfig(
        interpolator=NamedInterpolator.CURLY_BRACES,
        variable_escaper=Mock(spec=_noop_escaper),
        content_verifier=Mock(spec=_noop_verifier))


class TestRenderDriver:

    def test_simplest_happy_case(self):
        """A trivial template with only literal template strings must
        flatten them into an iterable of strings.
        """
        @template(fake_template_config, object())
        class FakeTemplate:
            ...

        def fake_prepopulate(render_context, *args, **kwargs):
            # Maybe slightly helpful for the test, mostly just for type hints
            assert isinstance(render_context, _RenderContext)

            render_context.template_preload[FakeTemplate] = (
                ParsedTemplateResource(
                    parts=(
                        LiteralTemplateString('foo', part_index=0),
                        LiteralTemplateString('bar', part_index=1),
                        LiteralTemplateString('baz', part_index=2),),
                    variable_names=frozenset(),
                    content_names=frozenset(),
                    slot_names=frozenset(),
                    slots={},
                    function_names=frozenset(),
                    function_calls={}))
            return
            yield

        with patch.object(
            _RenderContext,
            'prepopulate',
            side_effect=fake_prepopulate,
            autospec=True,
        ):
            parts_output = []
            error_collector = []
            __ = [
                *render_driver(FakeTemplate(), parts_output, error_collector)]

        assert parts_output == ['foo', 'bar', 'baz']
        assert not error_collector

    def test_with_precall(self):
        """A template that includes interpolated function calls must
        successfully gather their values from the render context
        precall.
        """
        # This can be anything, it just needs to be hashable and consistent
        fake_cache_key = 42

        @template(fake_template_config, object())
        class FakeTemplate:
            ...

        def fake_prepopulate(render_context, *args, **kwargs):
            # Maybe slightly helpful for the test, mostly just for type hints
            assert isinstance(render_context, _RenderContext)

            fake_interpolated_call = InterpolatedFunctionCall(
                call_args_exp=None,
                call_kwargs_exp=None,
                part_index=3,
                name='fakefunc',
                call_args=[],
                call_kwargs={})
            render_context.template_preload[FakeTemplate] = (
                ParsedTemplateResource(
                    parts=(
                        LiteralTemplateString('foo', part_index=0),
                        LiteralTemplateString('bar', part_index=1),
                        LiteralTemplateString('baz', part_index=2),
                        fake_interpolated_call),
                    variable_names=frozenset(),
                    content_names=frozenset(),
                    slot_names=frozenset(),
                    slots={},
                    function_names=frozenset({'fakefunc'}),
                    function_calls={'fakefunc': (fake_interpolated_call,)}))
            render_context.function_precall[fake_cache_key] = (
                FuncExecutionResult(
                    name='fakefunc', retval=('funky',), exc=None))
            return
            yield

        with patch.object(
            _RenderContext,
            'prepopulate',
            side_effect=fake_prepopulate,
            autospec=True,
        ), patch(
            'templatey.renderer._get_precall_cache_key',
            autospec=True,
            return_value=fake_cache_key
        ):
            parts_output = []
            error_collector = []
            __ = [
                *render_driver(FakeTemplate(), parts_output, error_collector)]

        assert parts_output == ['foo', 'bar', 'baz', 'funky']
        assert not error_collector

    def test_with_multiple_exceptions(self):
        """When rendering a template with exceptions, the render driver
        must collect all of these exceptions into a single exception
        group at the end of the driving.
        """
        @template(zderr_template_config, object())
        class FakeTemplate:
            var1: Var[str]

        def fake_prepopulate(render_context, *args, **kwargs):
            # Maybe slightly helpful for the test, mostly just for type hints
            assert isinstance(render_context, _RenderContext)

            render_context.template_preload[FakeTemplate] = (
                ParsedTemplateResource(
                    parts=(
                        LiteralTemplateString('foo', part_index=0),
                        LiteralTemplateString('bar', part_index=1),
                        LiteralTemplateString('baz', part_index=2),
                        InterpolatedVariable(
                            part_index=3,
                            name='var1',
                            config=InterpolationConfig()),
                        InterpolatedVariable(
                            part_index=4,
                            name='var1',
                            config=InterpolationConfig()),),
                    variable_names=frozenset({'var1'}),
                    content_names=frozenset(),
                    slot_names=frozenset(),
                    slots={},
                    function_names=frozenset(),
                    function_calls={}))
            return
            yield

        with patch.object(
            _RenderContext,
            'prepopulate',
            side_effect=fake_prepopulate,
            autospec=True,
        ):
            parts_output = []
            error_collector = []
            __ = [
                *render_driver(
                    FakeTemplate(var1='1var'),
                    parts_output,
                    error_collector)]

        assert parts_output == ['foo', 'bar', 'baz']
        assert len(error_collector) == 2


class TestRenderContext:

    def test_plain_template(self):
        """A plain template with no function calls or slots must
        generate a single RenderEnvRequest for the root template.
        """
        @template(fake_template_config, object())
        class FakeTemplate:
            var1: Var[str]

        ctx = _RenderContext(
            template_preload={},
            function_precall={},
            error_collector=[])

        request_count = 0
        for request in ctx.prepopulate(FakeTemplate(var1='foo')):
            request_count += 1

            assert not request.to_execute
            assert len(request.to_load) == 1
            assert FakeTemplate in request.to_load

            request.results_loaded[FakeTemplate] = ParsedTemplateResource(
                parts=(),
                variable_names=frozenset({'var1'}),
                content_names=frozenset(),
                slot_names=frozenset(),
                slots={},
                function_names=frozenset(),
                function_calls={})

        assert request_count == 1

    def test_template_with_func(self):
        """A template with one function call must generate two
        RenderEnvRequests: one for the root template and then one for
        the function.
        """
        @template(fake_template_config, object())
        class FakeTemplate:
            var1: Var[str]

        ctx = _RenderContext(
            template_preload={},
            function_precall={},
            error_collector=[])
        fake_interpolated_call = InterpolatedFunctionCall(
            call_args_exp=None,
            call_kwargs_exp=None,
            part_index=2,
            name='fakefunc',
            call_args=[],
            call_kwargs={})

        request_count = 0
        for request in ctx.prepopulate(FakeTemplate(var1='foo')):
            request_count += 1

            if request.to_load:
                assert not request.to_execute
                assert len(request.to_load) == 1
                assert FakeTemplate in request.to_load

                request.results_loaded[FakeTemplate] = ParsedTemplateResource(
                    parts=(
                        InterpolatedVariable(
                            part_index=0,
                            name='var1',
                            config=InterpolationConfig()),
                        LiteralTemplateString('baz', part_index=1),
                        fake_interpolated_call),
                    variable_names=frozenset({'var1'}),
                    content_names=frozenset(),
                    slot_names=frozenset(),
                    slots={},
                    function_names=frozenset({'fakefunc'}),
                    function_calls={'fakefunc': (fake_interpolated_call,)})

            else:
                # Note that this is just because we're on the first level of
                # function. Deeper nesting could result in both to_load and
                # to_execute being defined on the same request.
                assert not request.to_load
                assert len(request.to_execute) == 1
                exe_req, = request.to_execute

                request.results_executed[exe_req.result_key] = (
                    FuncExecutionResult(
                        name=exe_req.name, retval=('foo'), exc=None))

        assert request_count == 2

    def test_template_with_func_with_expansion(self):
        """Prepopulation with a function including arg/kwarg expansion
        must include those args and kwargs within the resulting
        call_args and call_kwargs for the execution request.
        """
        @template(fake_template_config, object())
        class FakeTemplate:
            var1: Var[str]

        ctx = _RenderContext(
            template_preload={},
            function_precall={},
            error_collector=[])
        fake_interpolated_call = InterpolatedFunctionCall(
            call_args_exp=['oof'],
            call_kwargs_exp={'baz': 'zab'},
            part_index=2,
            name='fakefunc',
            call_args=['foo'],
            call_kwargs={'bar': 'rab'})

        exe_req = None
        request_count = 0
        for request in ctx.prepopulate(FakeTemplate(var1='foo')):
            request_count += 1

            if request.to_load:
                assert not request.to_execute
                assert len(request.to_load) == 1
                assert FakeTemplate in request.to_load

                request.results_loaded[FakeTemplate] = ParsedTemplateResource(
                    parts=(fake_interpolated_call,),
                    variable_names=frozenset({'var1'}),
                    content_names=frozenset(),
                    slot_names=frozenset(),
                    slots={},
                    function_names=frozenset({'fakefunc'}),
                    function_calls={'fakefunc': (fake_interpolated_call,)})

            else:
                # Note that this is just because we're on the first level of
                # function. Deeper nesting could result in both to_load and
                # to_execute being defined on the same request.
                assert not request.to_load
                assert len(request.to_execute) == 1
                exe_req, = request.to_execute

                request.results_executed[exe_req.result_key] = (
                    FuncExecutionResult(
                        name=exe_req.name, retval=('foo'), exc=None))

        assert request_count == 2
        assert exe_req is not None
        assert exe_req.args == ('foo', 'oof')
        assert exe_req.kwargs == {'bar': 'rab', 'baz': 'zab'}


class TestRecursivelyCoerceFuncExecutionParams:
    """_recursively_coerce_func_execution_params()"""

    def test_int(self):
        """Integers must not break things, and must be returned
        unchanged.
        """
        retval = _recursively_coerce_func_execution_params(
            42,
            unescaped_vars=cast(_ParamLookup, {}),
            unverified_content=cast(_ParamLookup, {}))
        assert retval == 42

    def test_string(self):
        """Strings must return the string unchanged. In particular, they
        must not be expanded into a list of substrings, each one char
        long!
        """
        retval = _recursively_coerce_func_execution_params(
            'foo',
            unescaped_vars=cast(_ParamLookup, {}),
            unverified_content=cast(_ParamLookup, {}))
        assert retval == 'foo'

    def test_list_of_strings(self):
        """List of strings must also be returned unchanged, other than
        being coerced into a tuple.
        """
        retval = _recursively_coerce_func_execution_params(
            ['foo', 'bar'],
            unescaped_vars=cast(_ParamLookup, {}),
            unverified_content=cast(_ParamLookup, {}))
        assert retval == ('foo', 'bar')

    def test_dict_of_strings(self):
        """Dict of strings must also be returned unchanged
        """
        retval = _recursively_coerce_func_execution_params(
            {'foo': 'oof', 'bar': 'rab'},
            unescaped_vars=cast(_ParamLookup, {}),
            unverified_content=cast(_ParamLookup, {}))
        assert retval == {'foo': 'oof', 'bar': 'rab'}

    @pytest.mark.parametrize(
        'before,expected_after',
        [
            (NestedContentReference('foo'), 'oof'),
            ([NestedContentReference('foo')], ('oof',)),
            ({'foo': NestedContentReference('foo')}, {'foo': 'oof'}),
            (['beep', NestedContentReference('foo')], ('beep', 'oof')),
            (NestedVariableReference('bar'), 'rab'),
            ([NestedVariableReference('bar')], ('rab',)),
            ({'bar': NestedVariableReference('bar')}, {'bar': 'rab'}),
            (['beep', NestedVariableReference('bar')], ('beep', 'rab')),
            ([NestedContentReference('foo'), NestedVariableReference('bar')],
                ('oof', 'rab'))])
    def test_recursive_nested_reference(self, before, expected_after):
        """``NestedContentReference``s and ``NestedVariableReference``s,
        including those nested inside collections, must correctly be
        coerced (dereferenced).
        """
        fake_unverified_content = cast(_ParamLookup, {'foo': 'oof'})
        fake_unescaped_vars = cast(_ParamLookup, {'bar': 'rab'})
        retval = _recursively_coerce_func_execution_params(
            before,
            unverified_content=fake_unverified_content,
            unescaped_vars=fake_unescaped_vars)
        assert retval == expected_after


_testdata_apply_format = [
    ('foo', None, 'foo'),
    (1, None, '1'),
    (Decimal(1), None, '1'),
    (1, InterpolationConfig(fmt='02d'), "01"),
    ('foo', InterpolationConfig(fmt='_<14'), "foo___________"),
]


class TestApplyFormat:

    @pytest.mark.parametrize('raw,config,expected', _testdata_apply_format)
    def test_nones(self, raw, config, expected):
        rv = _apply_format(raw, config)
        assert rv == expected


class TestCaptureTraceback:

    def test_no_context(self):
        """Capturing tracebacks with no passed from_exc cause must
        correctly add a traceback.
        """
        exc = ZeroDivisionError('foo')
        assert exc.__traceback__ is None
        assert exc.__cause__ is None
        re_exc = _capture_traceback(exc)
        assert re_exc is exc
        assert re_exc.__traceback__ is not None
        assert re_exc.__cause__ is None

    def test_with_cause(self):
        """Capturing tracebacks with a passed from_exc cause must
        correctly add a traceback AND cause.
        """
        exc = ZeroDivisionError('foo')
        context = ZeroDivisionError('bar')
        assert exc.__traceback__ is None
        assert exc.__cause__ is None
        re_exc = _capture_traceback(exc, from_exc=context)
        assert re_exc is exc
        assert re_exc.__traceback__ is not None
        assert re_exc.__cause__ is not None


@patch(
    'templatey.renderer._apply_format',
    autospec=True,
    wraps=lambda raw_value, *_, **__: raw_value)
class TestCoerceInjectedValue:

    def test_escaped(self, apply_format_mock, new_fake_template_config):
        """An injected value that needs escaping must be escaped.
        """
        new_fake_template_config.variable_escaper.return_value = 'foobar'
        retval = _coerce_injected_value(
            InjectedValue(
                value='foo',
                use_variable_escaper=True,
                use_content_verifier=False),
            new_fake_template_config)
        assert apply_format_mock.call_count == 1
        assert retval == 'foobar'
        assert new_fake_template_config.variable_escaper.call_count == 1
        assert new_fake_template_config.content_verifier.call_count == 0

    def test_verified(self, apply_format_mock, new_fake_template_config):
        """An injected value that needs content verification must be
        verified.
        """
        new_fake_template_config.variable_escaper.return_value = 'foobar'
        retval = _coerce_injected_value(
            InjectedValue(
                value='foo',
                config=InterpolationConfig(),
                use_variable_escaper=False,
                use_content_verifier=True),
            new_fake_template_config)
        assert apply_format_mock.call_count == 1
        assert retval == 'foo'
        assert new_fake_template_config.variable_escaper.call_count == 0
        assert new_fake_template_config.content_verifier.call_count == 1

    def test_escaped_and_verified(
            self, apply_format_mock, new_fake_template_config):
        """An injected value that needs escaping and verification must
        have both be called.
        """
        new_fake_template_config.variable_escaper.return_value = 'foobar'
        retval = _coerce_injected_value(
            InjectedValue(
                value='foo',
                config=InterpolationConfig(),
                use_variable_escaper=True,
                use_content_verifier=True),
            new_fake_template_config)
        assert apply_format_mock.call_count == 1
        assert retval == 'foobar'
        assert new_fake_template_config.variable_escaper.call_count == 1
        assert new_fake_template_config.content_verifier.call_count == 1

    def test_nochecks(self, apply_format_mock, new_fake_template_config):
        """An injected value that needs no checks must not perform them.
        """
        new_fake_template_config.variable_escaper.return_value = 'foobar'
        retval = _coerce_injected_value(
            InjectedValue(
                value='foo',
                config=InterpolationConfig(),
                use_variable_escaper=False,
                use_content_verifier=False),
            new_fake_template_config)
        assert apply_format_mock.call_count == 1
        assert retval == 'foo'
        assert new_fake_template_config.variable_escaper.call_count == 0
        assert new_fake_template_config.content_verifier.call_count == 0
