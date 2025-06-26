from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import InitVar
from typing import cast

from templatey._types import TemplateIntersectable
from templatey._types import TemplateParamsInstance
from templatey.interpolators import NamedInterpolator
from templatey.parser import InterpolationConfig
from templatey.parser import ParsedTemplateResource
from templatey.templates import TemplateConfig
from templatey.templates import template


class _VirtualSlotMixin:
    """We use virtual slots on empty template instances to create
    provenance / render frame backstops for environment functions
    that return template instances. This mixin adds a slot to the
    class that allows it to be stored there.
    """
    _TEMPLATEY_VIRTUAL_INSTANCE = True
    _templatey_virtual_slots: Mapping[str, Sequence[TemplateParamsInstance]]
    __slots__ = ['_templatey_virtual_slots']


@template(
    TemplateConfig(
        interpolator=NamedInterpolator.UNICODE_CONTROL,
        variable_escaper=lambda value: value,
        content_verifier=lambda value: True),
    object()
)
class EmptyTemplate(_VirtualSlotMixin):
    """This is used as the render stack anchor for values that are
    injected into a function, and therefore have no parent. It is
    special-cased within the render env.
    """
    virtual_slots: InitVar[Mapping[str, Sequence[TemplateParamsInstance]]]

    def __post_init__(
            self,
            virtual_slots: Mapping[str, Sequence[TemplateParamsInstance]]):
        self._templatey_virtual_slots = virtual_slots

    def __getattr__(self, key: str):
        result = self._templatey_virtual_slots.get(key)
        if result is None:
            raise AttributeError(key)
        else:
            return result


PARSED_EMPTY_TEMPLATE = ParsedTemplateResource(
    parts=(),
    variable_names=frozenset(),
    content_names=frozenset(),
    slot_names=frozenset(),
    slots={},
    function_names=frozenset(),
    function_calls={})
EMPTY_TEMPLATE_XABLE = cast(type[TemplateIntersectable], EmptyTemplate)
EMPTY_TEMPLATE_INSTANCE = EmptyTemplate(virtual_slots={})
EMPTY_INTERPOLATION_CONFIG = InterpolationConfig()
