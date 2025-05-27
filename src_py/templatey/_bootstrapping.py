from __future__ import annotations

from typing import cast

from templatey.interpolators import NamedInterpolator
from templatey.parser import ParsedTemplateResource
from templatey.templates import TemplateConfig
from templatey.templates import TemplateIntersectable
from templatey.templates import template


@template(
    TemplateConfig(
        interpolator=NamedInterpolator.UNICODE_CONTROL,
        variable_escaper=lambda value: value,
        content_verifier=lambda value: True),
    object()
)
class EmptyTemplate:
    """This is used as the render stack anchor for values that are
    injected into a function, and therefore have no parent. It is
    special-cased within the render env.
    """


PARSED_EMPTY_TEMPLATE = ParsedTemplateResource(
    parts=(),
    variable_names=frozenset(),
    content_names=frozenset(),
    slot_names=frozenset(),
    slots={},
    function_names=frozenset(),
    function_calls={})
EMPTY_TEMPLATE_XABLE = cast(type[TemplateIntersectable], EmptyTemplate)
EMPTY_TEMPLATE_INSTANCE = EmptyTemplate()
