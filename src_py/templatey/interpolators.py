import re
from enum import Enum

_UNICODE_SGA = '\u0096'
_UNICODE_EGA = '\u0097'
# Printable as ␎
_UNICODE_SO = '\u240E'
# Printable as ␏
_UNICODE_SI = '\u240F'
_PATTERN_SUB_IN = re.compile(r'\{|\}|' + f'{_UNICODE_SO}|{_UNICODE_SI}')
_PATTERN_SUB_OUT = re.compile(f'{_UNICODE_SGA}|{_UNICODE_EGA}')


class NamedInterpolator(Enum):
    CURLY_BRACES = object()
    UNICODE_CONTROL = object()


def transform_unicode_control(text: str) -> str:
    """Python string formatting is implemented at the C level, which we
    don't want to mess with. So, although it's slightly less performant,
    when we want to use different interpolators, we apply a series of
    string transformations so that we can use the plain old python
    string formatting.

    Specifically, in the transform direction, we:
    1.. replace all ``{`` characters with ``U+0096``, "start of guarded
        area"
    2.. replace all ``}`` characters with ``U+0097``, "end of guarded
        area"
    3.. replace all ``␎`` (``U+240E``, "shift out") characters with
        ``{``
    4.. replace all ``␏`` (``U+240F``, "shift in") characters with ``}``
    """
    return _PATTERN_SUB_IN.sub(_transform_subber, text)


def untransform_unicode_control(text: str) -> str:
    """Python string formatting is implemented at the C level, which we
    don't want to mess with. So, although it's slightly less performant,
    when we want to use different interpolators, we apply a series of
    string transformations so that we can use the plain old python
    string formatting.

    Specifically, in the untransform direction, we:
    1.. replace ``U+0096`` ("start of guarded area") characters with
        ``{``
    2.. replace ``U+0097`` ("end of guarded area") characters with ``}``
    """
    return _PATTERN_SUB_OUT.sub(_untransform_subber, text)


def _transform_subber(match: re.Match) -> str:
    match_char = match.group(0)
    if match_char == _UNICODE_SO:
        return '{'
    elif match_char == _UNICODE_SI:
        return '}'
    elif match_char == '{':
        return _UNICODE_SGA
    elif match_char == '}':
        return _UNICODE_EGA
    else:
        raise ValueError('impossible branch: unknown subber match!')


def _untransform_subber(match: re.Match) -> str:
    match_char = match.group(0)
    if match_char == _UNICODE_SGA:
        return '{'
    elif match_char == _UNICODE_EGA:
        return '}'
    else:
        raise ValueError('impossible branch: unknown subber match!')
