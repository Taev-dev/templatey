from decimal import Decimal
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from templatey.environments import RenderEnvironment
from templatey.interpolators import NamedInterpolator
from templatey.parser import InterpolatedContent
from templatey.parser import InterpolatedFunctionCall
from templatey.parser import InterpolatedSlot
from templatey.parser import InterpolatedVariable
from templatey.parser import NestedContentReference
from templatey.parser import NestedVariableReference
from templatey.renderer import FuncExecutionRequest
from templatey.renderer import FuncExecutionResult
from templatey.renderer import _apply_format
from templatey.renderer import _capture_traceback
from templatey.renderer import _coerce_injected_value
from templatey.renderer import _recursively_coerce_func_execution_params
from templatey.templates import InjectedValue
from templatey.templates import TemplateConfig
from templatey.templates import template
from tests_py._utils import FakeComplexContent
from tests_py._utils import fake_template_config


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


class TestRenderDriverSync:
    """render_driver_sync()
    """

    def test_simplest_happy_case(self):
        """A trivial template with only strings must flatten them into
        a single iterator.
        """
        fake_render_env = Mock(spec=RenderEnvironment)

        @template(fake_template_config, object())
        class FakeTemplate:
            ...

        def inject_instead_of_flatten(template_instance, *, error_collector):
            yield 'foo'
            yield 'bar'
            yield 'baz'

        with patch(
            'templatey.renderer._flatten_and_interpolate',
            inject_instead_of_flatten
        ):
            parts = [*render_driver_sync(FakeTemplate(), fake_render_env)]

        assert parts == ['foo', 'bar', 'baz']

    def test_with_requests(self):
        """A template that issues help requests must be given answers in
        response, and the flattened result of those responses must be
        included in the final flattened output, without any missing
        items.

        This is meant to protect against improper generator driving, for
        example, by mixing .send() and a for loop.
        """
        fake_render_env = Mock(spec=RenderEnvironment)

        @template(fake_template_config, object())
        class FakeTemplate:
            ...

        def inject_instead_of_flatten(template_instance, *, error_collector):
            val1 = yield FakeTemplate()
            yield val1
            val2 = yield FuncExecutionRequest(name='foo', args=(), kwargs={})
            yield val2
            yield 'baz'

        fake_render_env.load_sync.return_value = 'foo'
        fake_render_env.execute_template_function_sync.return_value = 'bar'

        with patch(
            'templatey.renderer._flatten_and_interpolate',
            inject_instead_of_flatten
        ):
            parts = [*render_driver_sync(FakeTemplate(), fake_render_env)]

        assert parts == ['foo', 'bar', 'baz']
        assert fake_render_env.load_sync.call_count == 1
        assert fake_render_env.execute_template_function_sync.call_count == 1

    def test_with_multiple_exceptions(self):
        """When rendering a template with exceptions, the render driver
        must collect all of these exceptions into a single exception
        group at the end of the driving.
        """
        fake_render_env = Mock(spec=RenderEnvironment)

        @template(fake_template_config, object())
        class FakeTemplate:
            ...

        def inject_instead_of_flatten(template_instance, *, error_collector):
            yield 'foo'
            error_collector.append(ZeroDivisionError('bar'))
            yield 'baz'
            error_collector.append(ZeroDivisionError('zab'))

        with pytest.raises(ExceptionGroup) as exc_info:
            with patch(
                'templatey.renderer._flatten_and_interpolate',
                inject_instead_of_flatten
            ):
                __ = [*render_driver_sync(FakeTemplate(), fake_render_env)]

        raised = exc_info.value
        assert isinstance(raised, ExceptionGroup)
        assert len(raised.exceptions) == 2
        assert all(
            isinstance(exc, ZeroDivisionError) for exc in raised.exceptions)

    def test_with_single_exception(self):
        """When rendering a template with exceptions, the render driver
        must collect all of these exceptions into a single exception
        group at the end of the driving.

        This must also be true if the template had only a single
        exception (ie, it must never unwrap the exception and raise
        it directly).
        """
        fake_render_env = Mock(spec=RenderEnvironment)

        @template(fake_template_config, object())
        class FakeTemplate:
            ...

        def inject_instead_of_flatten(template_instance, *, error_collector):
            error_collector.append(ZeroDivisionError('bar'))
            yield 'foo'

        with pytest.raises(ExceptionGroup) as exc_info:
            with patch(
                'templatey.renderer._flatten_and_interpolate',
                inject_instead_of_flatten
            ):
                __ = [*render_driver_sync(FakeTemplate(), fake_render_env)]

        raised = exc_info.value
        assert isinstance(raised, ExceptionGroup)
        assert len(raised.exceptions) == 1
        assert all(
            isinstance(exc, ZeroDivisionError) for exc in raised.exceptions)

    def test_with_loading_exception(self):
        """When receiving a template loading request that fails, the
        exception raised must be added into the errors that are then
        raised as part of the ExceptionGroup.
        """
        fake_render_env = Mock(spec=RenderEnvironment)

        @template(fake_template_config, object())
        class FakeTemplate:
            ...

        def inject_instead_of_flatten(template_instance, *, error_collector):
            yield FakeTemplate()

        fake_render_env.load_sync.side_effect = ZeroDivisionError('foo')

        with pytest.raises(ExceptionGroup) as exc_info:
            with patch(
                'templatey.renderer._flatten_and_interpolate',
                inject_instead_of_flatten
            ):
                __ = [*render_driver_sync(FakeTemplate(), fake_render_env)]

        assert fake_render_env.load_sync.call_count == 1
        assert fake_render_env.execute_template_function_sync.call_count == 0
        raised = exc_info.value
        assert isinstance(raised, ExceptionGroup)
        assert len(raised.exceptions) == 1
        assert all(
            isinstance(exc, ZeroDivisionError) for exc in raised.exceptions)


class TestFlattenAndInterpolate:
    """_flatten_and_interpolate()"""

    @pytest.mark.skip
    def test_missing_slot_values(self):
        """Flattening must raise if a template instance is passed that
        still has a literal ellipsis as a value.
        """
        raise NotImplementedError(
            """
            TODO LEFT OFF HERE
            FIRST DO A GIT COMMIT!!!!
            Should decide re: testing these.
            First, may want to do the refactor you wanted re: moving error
            handling into flatten and interpolate, so that it's easier to
            do the async/sync driver.
            Also need to do the async driver!
            """)

    @pytest.mark.skip
    def test_var_precedence_parent_child(self):
        """Explicit variables passed in via the parent template's text
        must overwrite the value in the child template instance.
        """

    @pytest.mark.skip
    def test_error_loading_template(self):
        """If the requested template cannot be loaded successfully,
        flatten and interpolate must return early without raising
        additional errors, allowing the calling function to collect and
        re-raise the template loading failure.
        """


@patch(
    'templatey.renderer._apply_format',
    autospec=True,
    wraps=lambda raw_value, conversion, format_spec: raw_value)
class TestCoerceInterpolation:
    """_coerce_interpolation()"""

    def test_string(self, apply_format_mock, new_fake_template_config):
        """Strings must be returned unchanged and without formatting."""
        new_fake_template_config.variable_escaper.side_effect = _noop_escaper
        new_fake_template_config.content_verifier.side_effect = _noop_verifier
        retval, = _coerce_interpolation(
            'foo',
            new_fake_template_config,
            unescaped_vars={},
            unverified_content={},
            all_slots={},
            error_collector=[])

        assert apply_format_mock.call_count == 0
        assert new_fake_template_config.variable_escaper.call_count == 0
        assert new_fake_template_config.content_verifier.call_count == 0
        assert retval == 'foo'

    def test_interpolated_variable(
            self, apply_format_mock, new_fake_template_config):
        """Interpolated variables must have their formatting applied
        and then be escaped.
        """
        new_fake_template_config.variable_escaper.side_effect = _noop_escaper
        new_fake_template_config.content_verifier.side_effect = _noop_verifier
        retval, = _coerce_interpolation(
            InterpolatedVariable(
                name='foo', format_spec=None, conversion=None),
            new_fake_template_config,
            unescaped_vars={'foo': 'oof'},
            unverified_content={},
            all_slots={},
            error_collector=[])

        assert apply_format_mock.call_count == 1
        assert new_fake_template_config.variable_escaper.call_count == 1
        assert new_fake_template_config.content_verifier.call_count == 0
        assert retval == 'oof'

    def test_interpolated_content_simple(
            self, apply_format_mock, new_fake_template_config):
        """Interpolated content (of the simple text variety) must be
        substituted and verified.
        """
        new_fake_template_config.variable_escaper.side_effect = _noop_escaper
        new_fake_template_config.content_verifier.side_effect = _noop_verifier
        retval, = _coerce_interpolation(
            InterpolatedContent(name='foo'),
            new_fake_template_config,
            unescaped_vars={},
            unverified_content={'foo': 'oof'},
            all_slots={},
            error_collector=[])

        assert apply_format_mock.call_count == 0
        assert new_fake_template_config.variable_escaper.call_count == 0
        assert new_fake_template_config.content_verifier.call_count == 1
        assert retval == 'oof'

    def test_interpolated_content_complex(
            self, apply_format_mock, new_fake_template_config):
        """Interpolated content (of the ComplexContent variety) must be
        flattened, substituted, and verified.
        """
        new_fake_template_config.variable_escaper.side_effect = _noop_escaper
        new_fake_template_config.content_verifier.side_effect = _noop_verifier
        retval = [*_coerce_interpolation(
            InterpolatedContent(name='foo'),
            new_fake_template_config,
            unescaped_vars={'dog_count': 2, 'cat_count': 1},
            unverified_content={'foo': FakeComplexContent('dog_count', 'dog')},
            all_slots={},
            error_collector=[])]

        assert apply_format_mock.call_count == 1
        assert new_fake_template_config.variable_escaper.call_count == 1
        assert new_fake_template_config.content_verifier.call_count == 1
        assert retval == [2, ' dogs']

    def test_interpolated_slot(
            self, apply_format_mock, new_fake_template_config):
        """Interpolated slots must be recursed into and flattened.
        """
        new_fake_template_config.variable_escaper.side_effect = _noop_escaper
        new_fake_template_config.content_verifier.side_effect = _noop_verifier

        def _fake_flatten_and_interpolate(*args, **kwargs):
            yield from ['foo', 'bar', 'baz']

        with patch(
            'templatey.renderer._flatten_and_interpolate',
            _fake_flatten_and_interpolate
        ):
            retval = [*_coerce_interpolation(
                InterpolatedSlot(name='foo', params={}),
                new_fake_template_config,
                unescaped_vars={},
                unverified_content={},
                # The actual values here are unused, because we've patched
                # the call to flatten_and_interpolate.
                all_slots={'foo': [..., ...]},
                error_collector=[])]

        assert retval == ['foo', 'bar', 'baz'] * 2
        assert apply_format_mock.call_count == 0
        assert new_fake_template_config.variable_escaper.call_count == 0
        assert new_fake_template_config.content_verifier.call_count == 0

    @patch(
        'templatey.renderer._recursively_coerce_func_execution_params',
        autospec=True,
        side_effect=lambda val, *args, **kwargs: val)
    @patch(
        'templatey.renderer._coerce_injected_value', autospec=True)
    @patch(
        'templatey.renderer._flatten_and_interpolate', autospec=True)
    def test_interpolated_function_call_string_return(
            self, flatten_mock, coerce_mock, recurse_param_mock,
            apply_format_mock, new_fake_template_config):
        """Interpolated function calls which return a string must be
        passed to the variable escaper.
        """
        new_fake_template_config.variable_escaper.side_effect = _noop_escaper
        new_fake_template_config.content_verifier.side_effect = _noop_verifier

        generator = _coerce_interpolation(
            InterpolatedFunctionCall(
                name='foo',
                call_args=['foo'],
                call_kwargs={'bar': 'rab'}),
            new_fake_template_config,
            unescaped_vars={},
            unverified_content={},
            all_slots={},
            error_collector=[])
        exe_request = generator.send(None)

        assert isinstance(exe_request, FuncExecutionRequest)
        retval = [
            generator.send(
                FuncExecutionResult(retval=['foo', 'bar'], exc=None))]
        retval.extend(generator)

        assert retval == ['foo', 'bar']
        assert apply_format_mock.call_count == 0
        assert coerce_mock.call_count == 0
        assert flatten_mock.call_count == 0
        assert new_fake_template_config.variable_escaper.call_count == 2
        assert new_fake_template_config.content_verifier.call_count == 0

    @patch(
        'templatey.renderer._recursively_coerce_func_execution_params',
        autospec=True,
        side_effect=lambda val, *args, **kwargs: val)
    @patch(
        'templatey.renderer._coerce_injected_value', autospec=True)
    @patch(
        'templatey.renderer._flatten_and_interpolate', autospec=True)
    def test_interpolated_function_call_injection_return(
            self, flatten_mock, coerce_mock, recurse_param_mock,
            apply_format_mock, new_fake_template_config):
        """Interpolated function calls which return an InjectedValue
        instance must be routed to the coercer.
        """
        new_fake_template_config.variable_escaper.side_effect = _noop_escaper
        new_fake_template_config.content_verifier.side_effect = _noop_verifier

        generator = _coerce_interpolation(
            InterpolatedFunctionCall(
                name='foo',
                call_args=['foo'],
                call_kwargs={'bar': 'rab'}),
            new_fake_template_config,
            unescaped_vars={},
            unverified_content={},
            all_slots={},
            error_collector=[])
        exe_request = generator.send(None)

        assert isinstance(exe_request, FuncExecutionRequest)
        coerce_mock.return_value = 'foo'

        retval = [
            generator.send(
                FuncExecutionResult(
                    retval=[InjectedValue(
                        'foo', format_spec=None, conversion=None)],
                exc=None))]
        retval.extend(generator)

        assert retval == ['foo']
        assert apply_format_mock.call_count == 0
        assert coerce_mock.call_count == 1
        assert flatten_mock.call_count == 0
        # Note: because we bypassed the injection coercion, which does the
        # checks for us!
        assert new_fake_template_config.variable_escaper.call_count == 0
        assert new_fake_template_config.content_verifier.call_count == 0

    @patch(
        'templatey.renderer._recursively_coerce_func_execution_params',
        autospec=True,
        side_effect=lambda val, *args, **kwargs: val)
    @patch(
        'templatey.renderer._coerce_injected_value', autospec=True)
    @patch(
        'templatey.renderer._flatten_and_interpolate', autospec=True)
    def test_interpolated_function_call_nested_return(
            self, flatten_mock, coerce_mock, recurse_param_mock,
            apply_format_mock, new_fake_template_config):
        """Interpolated function calls which return a nested template
        must be routed to _flatten_and_interpolate.
        """
        @template(new_fake_template_config, object())
        class FakeTemplate:
            ...

        new_fake_template_config.variable_escaper.side_effect = _noop_escaper
        new_fake_template_config.content_verifier.side_effect = _noop_verifier

        generator = _coerce_interpolation(
            InterpolatedFunctionCall(
                name='foo',
                call_args=['foo'],
                call_kwargs={'bar': 'rab'}),
            new_fake_template_config,
            unescaped_vars={},
            unverified_content={},
            all_slots={},
            error_collector=[])
        exe_request = generator.send(None)

        assert isinstance(exe_request, FuncExecutionRequest)
        def _fake_flatten_and_interpolate(*args, **kwargs):
            yield from ['foo', 'bar', 'baz']
        flatten_mock.return_value = _fake_flatten_and_interpolate()

        retval = [
            generator.send(
                FuncExecutionResult(
                    retval=[FakeTemplate()],
                exc=None))]
        retval.extend(generator)

        assert retval == ['foo', 'bar', 'baz']
        assert apply_format_mock.call_count == 0
        assert coerce_mock.call_count == 0
        assert flatten_mock.call_count == 1
        # Note: because we bypassed _flatten, which ultimately results in
        # these calls!
        assert new_fake_template_config.variable_escaper.call_count == 0
        assert new_fake_template_config.content_verifier.call_count == 0


class TestRecursivelyCoerceFuncExecutionParams:
    """_recursively_coerce_func_execution_params()"""

    def test_int(self):
        """Integers must not break things, and must be returned
        unchanged.
        """
        retval = _recursively_coerce_func_execution_params(
            42,
            unescaped_vars={},
            unverified_content={})
        assert retval == 42

    def test_string(self):
        """Strings must return the string unchanged. In particular, they
        must not be expanded into a list of substrings, each one char
        long!
        """
        retval = _recursively_coerce_func_execution_params(
            'foo',
            unescaped_vars={},
            unverified_content={})
        assert retval == 'foo'

    def test_list_of_strings(self):
        """List of strings must also be returned unchanged, other than
        being coerced into a tuple.
        """
        retval = _recursively_coerce_func_execution_params(
            ['foo', 'bar'],
            unescaped_vars={},
            unverified_content={})
        assert retval == ('foo', 'bar')

    def test_dict_of_strings(self):
        """Dict of strings must also be returned unchanged
        """
        retval = _recursively_coerce_func_execution_params(
            {'foo': 'oof', 'bar': 'rab'},
            unescaped_vars={},
            unverified_content={})
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
        retval = _recursively_coerce_func_execution_params(
            before,
            unverified_content={'foo': 'oof'},
            unescaped_vars={'bar': 'rab'})
        assert retval == expected_after


_testdata_apply_format = [
    ('foo', None, None, 'foo'),
    (1, None, None, '1'),
    (Decimal(1), None, None, '1'),
    (Decimal(1), 'r', None, "Decimal('1')"),
    (1, None, '02d', "01"),
    (Decimal(1), 'r', '_<14', "Decimal('1')__"),
]


class TestApplyFormat:

    @pytest.mark.parametrize(
        'raw,conversion,fmt_spec,expected', _testdata_apply_format)
    def test_nones(self, raw, conversion, fmt_spec, expected):
        rv = _apply_format(raw, conversion, fmt_spec)
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
    wraps=lambda raw_value, conversion, format_spec: raw_value)
class TestCoerceInjectedValue:

    def test_escaped(self, apply_format_mock, new_fake_template_config):
        """An injected value that needs escaping must be escaped.
        """
        new_fake_template_config.variable_escaper.return_value = 'foobar'
        retval = _coerce_injected_value(
            InjectedValue(
                value='foo',
                format_spec=None,
                conversion=None,
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
                format_spec=None,
                conversion=None,
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
                format_spec=None,
                conversion=None,
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
                format_spec=None,
                conversion=None,
                use_variable_escaper=False,
                use_content_verifier=False),
            new_fake_template_config)
        assert apply_format_mock.call_count == 1
        assert retval == 'foo'
        assert new_fake_template_config.variable_escaper.call_count == 0
        assert new_fake_template_config.content_verifier.call_count == 0
