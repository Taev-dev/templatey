from unittest.mock import Mock

from templatey.interpolators import NamedInterpolator
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
