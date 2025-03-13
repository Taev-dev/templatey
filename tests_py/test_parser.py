from templatey.interpolators import NamedInterpolator
from templatey.parser import InterpolatedContent
from templatey.parser import InterpolatedFunctionCall
from templatey.parser import InterpolatedSlot
from templatey.parser import InterpolatedVariable
from templatey.parser import NestedContentReference
from templatey.parser import NestedVariableReference
from templatey.parser import parse


# TODO: should also add more specific tests for the individual private
# functions used by parse
class TestParse:

    def test_unicodecc_with_variable_and_format_spec(self):
        template = 'foo {␎var.bar:04d␏}'
        parsed = parse(template, NamedInterpolator.UNICODE_CONTROL)

        assert len(parsed.parts) == 3
        assert parsed.parts[0] == 'foo {'
        assert parsed.parts[1] == InterpolatedVariable(
            part_index=1, name='bar', format_spec='04d', conversion=None)
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
        assert parsed.parts[1] == InterpolatedContent(part_index=1, name='bar')
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
            part_index=1, name='bar', format_spec=None, conversion=None)
        assert parsed.variable_names == frozenset({'bar'})
        assert not parsed.content_names
        assert not parsed.slot_names
        assert not parsed.function_names
        assert not parsed.function_calls

    def test_curlybrace_with_slot_simple(self):
        template = 'foo {slot.bar}'
        parsed = parse(template, NamedInterpolator.CURLY_BRACES)

        assert len(parsed.parts) == 2
        assert parsed.parts[0] == 'foo '
        assert parsed.parts[1] == InterpolatedSlot(
            part_index=1, name='bar', params={})
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
            part_index=1, name='bar', params={'baz': 'zab'})
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
            params={'baz': NestedContentReference(name='baz')})
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
        assert parsed.parts[1] == InterpolatedFunctionCall(
           part_index=1,  name='bar', call_args=[], call_kwargs={})
        assert parsed.parts[2] == ' '
        assert parsed.parts[3] == InterpolatedFunctionCall(
            part_index=3, name='bar', call_args=[], call_kwargs={})
        assert not parsed.slot_names
        assert not parsed.content_names
        assert not parsed.variable_names
        assert parsed.function_names == frozenset({'bar'})
        assert parsed.function_calls['bar'] == (
            InterpolatedFunctionCall(
                part_index=1, name='bar', call_args=[], call_kwargs={}),
            InterpolatedFunctionCall(
                part_index=3, name='bar', call_args=[], call_kwargs={}))

    def test_curlybrace_with_function_params_constant(self):
        template = 'foo {@bar(1, baz="zab")}'
        parsed = parse(template, NamedInterpolator.CURLY_BRACES)

        assert len(parsed.parts) == 2
        assert parsed.parts[0] == 'foo '
        assert parsed.parts[1] == InterpolatedFunctionCall(
            part_index=1,
            name='bar',
            call_args=[1],
            call_kwargs={'baz': 'zab'})
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
        assert parsed.parts[1] == InterpolatedFunctionCall(
            part_index=1,
            name='bar',
            call_args=[NestedVariableReference(name='baz')],
            call_kwargs={})
        assert not parsed.slot_names
        assert not parsed.content_names
        assert parsed.variable_names == frozenset({'baz'})
        assert parsed.function_names == frozenset({'bar'})
        assert 'bar' in parsed.function_calls
