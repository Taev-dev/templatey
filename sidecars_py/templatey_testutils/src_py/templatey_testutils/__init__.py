from collections.abc import Iterable
from unittest.mock import Mock

from templatey.interpolators import NamedInterpolator
from templatey.parser import InterpolatedVariable
from templatey.templates import TemplateConfig


def _variable_escaper_spec(value: str) -> str: ...


fake_template_config = TemplateConfig(
    interpolator=NamedInterpolator.CURLY_BRACES,
    variable_escaper=Mock(wraps=lambda value: value),
    content_verifier=Mock(wraps=lambda value: True))

zderr_template_config = TemplateConfig(
    interpolator=NamedInterpolator.CURLY_BRACES,
    variable_escaper=Mock(
        spec=_variable_escaper_spec, side_effect=ZeroDivisionError()),
    content_verifier=Mock(wraps=lambda value: True))


class FakeComplexContent:
    TEMPLATEY_CONTENT = True

    def __init__(self, keyword: str, noun: str):
        self._keyword = keyword
        self._noun = noun

    def flatten(
            self,
            unescaped_vars_context: dict[str, int],
            parent_part_index: int,
            ) -> Iterable[str | InterpolatedVariable]:
        """This is a very simple kind of complex content that... well,
        constructs a correct plural of the word dog based on the
        ``dog_count`` variable.
        """
        is_plural = unescaped_vars_context[self._keyword] > 1
        yield InterpolatedVariable(
            name=self._keyword, format_spec=None, conversion=None,
            part_index=parent_part_index)
        if is_plural:
            yield f' {self._noun}s'
        else:
            yield f' {self._noun}'
