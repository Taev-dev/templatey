from __future__ import annotations

import ast
import itertools
import re
import string
from collections import defaultdict
from collections.abc import Collection
from collections.abc import Generator
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from functools import singledispatch
from typing import cast

from templatey.exceptions import DuplicateSlotName
from templatey.exceptions import InvalidTemplateInterpolation
from templatey.interpolators import NamedInterpolator
from templatey.interpolators import transform_unicode_control
from templatey.interpolators import untransform_unicode_control

_SLOT_MATCHER = re.compile(r'^\s*slot\.([A-z_][A-z0-9_]*)\s*$')
_CONTENT_MATCHER = re.compile(r'^\s*content\.([A-z_][A-z0-9_]*)\s*$')
_VAR_MATCHER = re.compile(r'^\s*var\.([A-z_][A-z0-9_]*)\s*$')
_FUNC_MATCHER = re.compile(r'^\s*@([A-z_][A-z0-9_]*)\(([^\)]*)\)\s*$')


@dataclass(frozen=True)
class ParsedTemplateResource:
    """In addition to storing the actual template parts, this stores
    information about which references the template had, for use later
    when validating the template (within some render context).
    """
    parts: tuple[
        LiteralTemplateString
        | InterpolatedSlot
        | InterpolatedContent
        | InterpolatedVariable
        | InterpolatedFunctionCall, ...]
    variable_names: frozenset[str]
    content_names: frozenset[str]
    slot_names: frozenset[str]
    function_names: frozenset[str]
    # Separate this out from function_names so that we can put compare=False
    # while still preserving comparability between instances. It's not clear if
    # this is useful, but the memory footprint should be low
    # Note: this is included for convenience, so that the render environment
    # has easy access to all of the *args and **kwargs, so that they can be
    # tested against the signature of the actual render function during loading
    function_calls: dict[str, tuple[InterpolatedFunctionCall]] = field(
        compare=False)
    slots: dict[str, InterpolatedSlot] = field(compare=False)


class LiteralTemplateString(str):
    __slots__ = ['part_index']
    part_index: int

    def __new__(cls, *args, part_index: int, **kwargs):
        instance = super().__new__(cls, *args, **kwargs)
        instance.part_index = part_index
        return instance


@dataclass(frozen=True, slots=True)
class InterpolatedContent:
    part_index: int
    # TODO: this needs a way to define any variables used by the content via
    # ComplexContent! Otherwise, strict mode in template/interface validation
    # will always fail with interpolated content.
    name: str


@dataclass(frozen=True, slots=True)
class InterpolatedSlot:
    part_index: int
    name: str
    params: dict[str, object]


@dataclass(frozen=True, slots=True)
class InterpolatedVariable:
    part_index: int
    name: str
    format_spec: str | None
    conversion: str | None


@dataclass(frozen=True, slots=True)
class InterpolatedFunctionCall:
    part_index: int
    name: str
    call_args: list[object] = field(compare=False)
    call_kwargs: dict[str, object] = field(compare=False)


@dataclass(slots=True, frozen=True)
class NestedContentReference:
    name: str


@dataclass(slots=True, frozen=True)
class NestedVariableReference:
    name: str


_VALID_NESTED_REFS = {
    'content': NestedContentReference,
    'var': NestedVariableReference}


def parse(
        template_str: str,
        interpolator: NamedInterpolator
        ) -> ParsedTemplateResource:
    if interpolator is NamedInterpolator.UNICODE_CONTROL:
        do_untransform = True
        template_str = transform_unicode_control(template_str)
    else:
        do_untransform = False

    parts = tuple(
        _wrap_formatter_parse(template_str, do_untransform=do_untransform))

    content_names = set()
    slot_names = set()
    variable_names = set()
    functions = defaultdict(list)
    for part in parts:
        if isinstance(part, InterpolatedContent):
            content_names.add(part.name)
        elif isinstance(part, InterpolatedVariable):
            variable_names.add(part.name)
        elif isinstance(part, InterpolatedSlot):
            # Interpolated slots must be unique; enforce that here.
            if part.name in slot_names:
                raise DuplicateSlotName(part.name)

            for maybe_reference in part.params.values():
                nested_content_refs, nested_var_refs = _extract_nested_refs(
                    maybe_reference)
                content_names.update(ref.name for ref in nested_content_refs)
                variable_names.update(ref.name for ref in nested_var_refs)

            slot_names.add(part.name)

        elif isinstance(part, InterpolatedFunctionCall):
            for maybe_reference in itertools.chain(
                part.call_args, part.call_kwargs.values()
            ):
                nested_content_refs, nested_var_refs = _extract_nested_refs(
                    maybe_reference)
                content_names.update(ref.name for ref in nested_content_refs)
                variable_names.update(ref.name for ref in nested_var_refs)

            functions[part.name].append(part)

    return ParsedTemplateResource(
        parts=parts,
        content_names=frozenset(content_names),
        variable_names=frozenset(variable_names),
        slot_names=frozenset(slot_names),
        function_names=frozenset(functions),
        function_calls={
            name: tuple(calls) for name, calls in functions.items()},
        slots={
            maybe_slot.name: maybe_slot
            for maybe_slot in parts
            if isinstance(maybe_slot, InterpolatedSlot)})


def _extract_nested_refs(value) -> tuple[
        set[NestedContentReference], set[NestedVariableReference]]:
    """Call this to recursively extract all of the content and variable
    references contained within an environment function call.
    """
    content_refs = set()
    var_refs = set()

    # Note that order here is important! Mappings are always collections!
    # Strings are always collections, too!
    if isinstance(value, Mapping):
        nested_values = value.values()
    elif isinstance(value, str):
        nested_values = ()
    elif isinstance(value, Collection):
        nested_values = value
    else:
        nested_values = ()

        if isinstance(value, NestedContentReference):
            content_refs.add(value)

        elif isinstance(value, NestedVariableReference):
            var_refs.add(value)

    for nested_val in nested_values:
        nested_content_refs, nested_var_refs = _extract_nested_refs(
            nested_val)
        content_refs.update(nested_content_refs)
        nested_var_refs.update(nested_var_refs)

    return content_refs, var_refs


def _wrap_formatter_parse(
        formattable_template_str: str,
        do_untransform=False
        ) -> Generator[
            LiteralTemplateString
                | InterpolatedSlot
                | InterpolatedContent
                | InterpolatedVariable
                | InterpolatedFunctionCall,
            None,
            None]:
    """A generator. Wraps the very weird API provided by
    string.Formatter.parse, instead yielding either:
    ++  literal text, in string format
    ++  ``InterpolatedContent`` instances
    ++  ``InterpolatedSlot`` instances
    ++  ``InterpolatedVariable`` instances
    ++  ``InterpolatedFunctionCall`` instances
    """
    part_counter = itertools.count()
    formatter = string.Formatter()

    for format_tuple in formatter.parse(formattable_template_str):
        # Arg order here is: literal_text, field_name, format_spec, conversion
        # Note that this can contain BOTH a literal text and a field name.
        # It's a really weird API; that's why we're wrapping it in
        # _extract_formatting_kwargs. It still reads things left-to-right, but
        # it bundles them together really strangely
        if do_untransform:
            literal_text, field_name, format_spec, conversion = (
                untransform_unicode_control(format_tuple_part)
                if format_tuple_part is not None else None
                for format_tuple_part in format_tuple)
        else:
            literal_text, field_name, format_spec, conversion = format_tuple

        if literal_text is not None:
            if do_untransform:
                yield LiteralTemplateString(
                    untransform_unicode_control(literal_text),
                    part_index=next(part_counter))
            else:
                yield LiteralTemplateString(
                    literal_text,
                    part_index=next(part_counter))

        # field_name can be None, an empty string, or a kwargname.
        # None means there's no formatting field left in the string -- in which
        # case, the literal_text would contain the rest of the string.
        if field_name is None:
            continue
        else:
            yield _coerce_interpolation(
                field_name, format_spec, conversion, part_counter)


def _coerce_interpolation(field_name, format_spec, conversion, part_counter):
    if (match := _VAR_MATCHER.match(field_name)) is not None:
        return InterpolatedVariable(
            part_index=next(part_counter),
            name=match.group(1),
            format_spec=format_spec or None,
            conversion=conversion or None)

    if conversion:
        raise InvalidTemplateInterpolation(
            'templatey only supports str conversions in var interpolations',
            field_name, format_spec, conversion)

    if (match := _SLOT_MATCHER.match(field_name)) is not None:
        slot_params_str = format_spec.strip()
        try:
            args, kwargs = _extract_call_signature(slot_params_str)

            if args:
                raise ValueError('Slot parameters are keyword-only!')

        except (ValueError, SyntaxError) as exc:
            raise InvalidTemplateInterpolation(
                'Invalid slot parameters!',
                field_name, format_spec, conversion) from exc

        return InterpolatedSlot(
            part_index=next(part_counter),
            name=match.group(1),
            params=kwargs)

    # The format spec is determined by the first : in the interpolation. Any
    # following :s are included as part of it. However, in the rest of these,
    # we want to interpret the format spec : as literally part of the
    # field_name, so we join them back up.
    if format_spec:
        full_interpolation_def = f'{field_name}:{format_spec}'
    else:
        full_interpolation_def = field_name

    if (match := _CONTENT_MATCHER.match(full_interpolation_def)) is not None:
        return InterpolatedContent(
            part_index=next(part_counter),
            name=match.group(1))

    if (match := _FUNC_MATCHER.match(full_interpolation_def)) is not None:
        try:
            args, kwargs = _extract_call_signature(match.group(2))
        except (ValueError, SyntaxError) as exc:
            raise InvalidTemplateInterpolation(
                'Invalid asset function call signature',
                field_name, format_spec, conversion) from exc
        return InterpolatedFunctionCall(
            part_index=next(part_counter),
            name=match.group(1),
            call_args=args,
            call_kwargs=kwargs)

    raise InvalidTemplateInterpolation(
        'Unknown target for templatey interpolation',
        field_name, format_spec, conversion)


def _extract_call_signature(str_signature):
    """Returns *args and **kwargs for the desired asset function."""
    tree = ast.parse(f'print({str_signature})')
    injected_print = cast(ast.Call, cast(ast.Expr, tree.body[0]).value)

    args = []
    kwargs = {}

    for ast_arg in injected_print.args:
        args.append(_extract_reference_or_literal(ast_arg))
    for ast_kwarg in injected_print.keywords:
        kwargs[ast_kwarg.arg] = _extract_reference_or_literal(ast_kwarg.value)

    return args, kwargs


@singledispatch
def _extract_reference_or_literal(ast_node):
    """Gets the actual reference out of an AST node used in the call
    signature, either as an arg or the value of a kwarg.
    """
    raise ValueError('No matching node type', ast_node)


@_extract_reference_or_literal.register
def _(ast_node: ast.Attribute):
    should_be_name = ast_node.value
    if not isinstance(should_be_name, ast.Name):
        raise ValueError('Non-literals must be attributes', ast_node)

    target_cls = _VALID_NESTED_REFS.get(should_be_name.id)
    if target_cls is None:
        raise ValueError(
            'Invalid asset reference value for asset function',
            should_be_name.id)

    return target_cls(name=ast_node.attr)


@_extract_reference_or_literal.register
def _(ast_node: ast.Constant):
    return ast_node.value
