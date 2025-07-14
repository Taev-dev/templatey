from templatey.interpolators import NamedInterpolator
from templatey.parser import InterpolatedContent
from templatey.parser import InterpolatedFunctionCall
from templatey.parser import InterpolatedSlot
from templatey.parser import InterpolatedVariable
from templatey.parser import InterpolationConfig
from templatey.parser import NestedContentReference
from templatey.parser import NestedVariableReference
from templatey.parser import parse


# TODO: should also add more specific tests for the individual private
# functions used by parse
class TestParse:

    def test_unicodecc_with_variable_and_format_spec(self):
        template = 'foo {␎var.bar:__fmt__="04d"␏}'
        parsed = parse(template, NamedInterpolator.UNICODE_CONTROL)

        assert len(parsed.parts) == 3
        assert parsed.parts[0] == 'foo {'
        assert parsed.parts[1] == InterpolatedVariable(
            part_index=1,
            name='bar',
            config=InterpolationConfig(fmt='04d'))
        assert parsed.parts[2] == '}'
        assert not parsed.content_names
        assert not parsed.slot_names
        assert parsed.variable_names == frozenset({'bar'})
        assert not parsed.function_names
        assert not parsed.function_calls

    def test_curlybrace_plain_string(self):
        template = 'foo'
        parsed = parse(template, NamedInterpolator.CURLY_BRACES)

        assert len(parsed.parts) == 1
        assert parsed.parts[0] == template
        assert not parsed.content_names
        assert not parsed.slot_names
        assert not parsed.variable_names
        assert not parsed.function_names
        assert not parsed.function_calls

    def test_curlybrace_with_content(self):
        template = 'foo {content.bar}'
        parsed = parse(template, NamedInterpolator.CURLY_BRACES)

        assert len(parsed.parts) == 2
        assert parsed.parts[0] == 'foo '
        assert parsed.parts[1] == InterpolatedContent(
            part_index=1,
            name='bar',
            config=InterpolationConfig())
        assert parsed.content_names == frozenset({'bar'})
        assert not parsed.slot_names
        assert not parsed.variable_names
        assert not parsed.function_names
        assert not parsed.function_calls

    def test_curlybrace_with_variable(self):
        template = 'foo {var.bar}'
        parsed = parse(template, NamedInterpolator.CURLY_BRACES)

        assert len(parsed.parts) == 2
        assert parsed.parts[0] == 'foo '
        assert parsed.parts[1] == InterpolatedVariable(
            part_index=1,
            name='bar',
            config=InterpolationConfig())
        assert parsed.variable_names == frozenset({'bar'})
        assert not parsed.content_names
        assert not parsed.slot_names
        assert not parsed.function_names
        assert not parsed.function_calls

    def test_curlybrace_with_variable_after_whitespace(self):
        template = 'foo {\n    var.bar}'
        parsed = parse(template, NamedInterpolator.CURLY_BRACES)

        assert len(parsed.parts) == 2
        assert parsed.parts[0] == 'foo '
        assert parsed.parts[1] == InterpolatedVariable(
            part_index=1,
            name='bar',
            config=InterpolationConfig())
        assert parsed.variable_names == frozenset({'bar'})
        assert not parsed.content_names
        assert not parsed.slot_names
        assert not parsed.function_names
        assert not parsed.function_calls

    def test_curlybrace_with_content_and_affix(self):
        template = r'foo {content.bar: __prefix__="\n", __suffix__=";"}'
        parsed = parse(template, NamedInterpolator.CURLY_BRACES)

        assert len(parsed.parts) == 2
        assert parsed.parts[0] == 'foo '
        assert parsed.parts[1] == InterpolatedContent(
            part_index=1,
            name='bar',
            config=InterpolationConfig(
                fmt=None,
                prefix='\n',
                suffix=';'))
        assert not parsed.variable_names
        assert parsed.content_names == frozenset({'bar'})
        assert not parsed.slot_names
        assert not parsed.function_names
        assert not parsed.function_calls

    def test_curlybrace_with_slot_simple(self):
        template = 'foo {slot.bar}'
        parsed = parse(template, NamedInterpolator.CURLY_BRACES)

        assert len(parsed.parts) == 2
        assert parsed.parts[0] == 'foo '
        assert parsed.parts[1] == InterpolatedSlot(
            part_index=1,
            name='bar',
            params={},
            config=InterpolationConfig())
        assert parsed.slot_names == frozenset({'bar'})
        assert not parsed.content_names
        assert not parsed.variable_names
        assert not parsed.function_names
        assert not parsed.function_calls

    def test_curlybrace_with_slot_params_constant(self):
        template = 'foo {slot.bar: baz="zab"}'
        parsed = parse(template, NamedInterpolator.CURLY_BRACES)

        assert len(parsed.parts) == 2
        assert parsed.parts[0] == 'foo '
        assert parsed.parts[1] == InterpolatedSlot(
            part_index=1,
            name='bar',
            params={'baz': 'zab'},
            config=InterpolationConfig())
        assert parsed.slot_names == frozenset({'bar'})
        assert not parsed.content_names
        assert not parsed.variable_names
        assert not parsed.function_names
        assert not parsed.function_calls

    def test_curlybrace_with_slot_params_content_ref(self):
        template = 'foo {slot.bar: baz=content.baz}'
        parsed = parse(template, NamedInterpolator.CURLY_BRACES)

        assert len(parsed.parts) == 2
        assert parsed.parts[0] == 'foo '
        assert parsed.parts[1] == InterpolatedSlot(
            part_index=1, name='bar',
            params={'baz': NestedContentReference(name='baz')},
            config=InterpolationConfig())
        assert parsed.slot_names == frozenset({'bar'})
        assert parsed.content_names == frozenset({'baz'})
        assert not parsed.variable_names
        assert not parsed.function_names
        assert not parsed.function_calls

    def test_curlybrace_with_function_simple_2x(self):
        template = 'foo {@bar()} {@bar()}'
        parsed = parse(template, NamedInterpolator.CURLY_BRACES)

        assert len(parsed.parts) == 4
        assert parsed.parts[0] == 'foo '
        assert InterpolatedFunctionCall(
            call_args_exp=None,
            call_kwargs_exp=None,
            part_index=1,
            name='bar',
            call_args=[],
            call_kwargs={})._matches(parsed.parts[1])
        assert parsed.parts[2] == ' '
        assert InterpolatedFunctionCall(
            call_args_exp=None,
            call_kwargs_exp=None,
            part_index=3,
            name='bar',
            call_args=[],
            call_kwargs={})._matches(parsed.parts[3])
        assert not parsed.slot_names
        assert not parsed.content_names
        assert not parsed.variable_names
        assert parsed.function_names == frozenset({'bar'})
        assert parsed.function_calls['bar'] == (
            InterpolatedFunctionCall(
                call_args_exp=None,
                call_kwargs_exp=None,
                part_index=1,
                name='bar',
                call_args=[],
                call_kwargs={}),
            InterpolatedFunctionCall(
                call_args_exp=None,
                call_kwargs_exp=None,
                part_index=3,
                name='bar',
                call_args=[],
                call_kwargs={}))

    def test_curlybrace_with_function_params_constant(self):
        template = 'foo {@bar(1, baz="zab")}'
        parsed = parse(template, NamedInterpolator.CURLY_BRACES)

        assert len(parsed.parts) == 2
        assert parsed.parts[0] == 'foo '
        assert InterpolatedFunctionCall(
            call_args_exp=None,
            call_kwargs_exp=None,
            part_index=1,
            name='bar',
            call_args=[1],
            call_kwargs={'baz': 'zab'})._matches(parsed.parts[1])
        assert not parsed.slot_names
        assert not parsed.content_names
        assert not parsed.variable_names
        assert parsed.function_names == frozenset({'bar'})
        assert 'bar' in parsed.function_calls

    def test_curlybrace_with_function_params_var_ref(self):
        template = 'foo {@bar(var.baz)}'
        parsed = parse(template, NamedInterpolator.CURLY_BRACES)

        assert len(parsed.parts) == 2
        assert parsed.parts[0] == 'foo '
        assert InterpolatedFunctionCall(
            call_args_exp=None,
            call_kwargs_exp=None,
            part_index=1,
            name='bar',
            call_args=[NestedVariableReference(name='baz')],
            call_kwargs={})._matches(parsed.parts[1])
        assert not parsed.slot_names
        assert not parsed.content_names
        assert parsed.variable_names == frozenset({'baz'})
        assert parsed.function_names == frozenset({'bar'})
        assert 'bar' in parsed.function_calls

    def test_curlybrace_with_function_params_var_ref_arg_expansion(self):
        template = 'foo {@bar(*var.baz)}'
        parsed = parse(template, NamedInterpolator.CURLY_BRACES)

        assert len(parsed.parts) == 2
        assert parsed.parts[0] == 'foo '
        assert InterpolatedFunctionCall(
            call_args_exp=NestedVariableReference(name='baz'),
            call_kwargs_exp=None,
            part_index=1,
            name='bar',
            call_args=[],
            call_kwargs={})._matches(parsed.parts[1])
        assert not parsed.slot_names
        assert not parsed.content_names
        assert parsed.variable_names == frozenset({'baz'})
        assert parsed.function_names == frozenset({'bar'})
        assert 'bar' in parsed.function_calls

    def test_curlybrace_with_function_params_var_ref_kwarg_expansion(self):
        template = 'foo {@bar(**var.baz)}'
        parsed = parse(template, NamedInterpolator.CURLY_BRACES)

        assert len(parsed.parts) == 2
        assert parsed.parts[0] == 'foo '
        assert InterpolatedFunctionCall(
            call_args_exp=None,
            call_kwargs_exp=NestedVariableReference(name='baz'),
            part_index=1,
            name='bar',
            call_args=[],
            call_kwargs={})._matches(parsed.parts[1])
        assert not parsed.slot_names
        assert not parsed.content_names
        assert parsed.variable_names == frozenset({'baz'})
        assert parsed.function_names == frozenset({'bar'})
        assert 'bar' in parsed.function_calls

    def test_curlybrace_with_comment(self):
        template = 'foo {# some comment} bar'
        parsed = parse(template, NamedInterpolator.CURLY_BRACES)

        assert len(parsed.parts) == 2
        assert parsed.parts[0] == 'foo '
        assert parsed.parts[1] == ' bar'
        assert not parsed.variable_names
        assert not parsed.content_names
        assert not parsed.slot_names
        assert not parsed.function_names
        assert not parsed.function_calls
