# import xml.sax.saxutils
from contextvars import ContextVar
from html import escape as html_escape
from html.parser import HTMLParser
from typing import Annotated
from typing import Literal

from docnote import ClcNote

from templatey.exceptions import BlockedContentValue
from templatey.interpolators import NamedInterpolator
from templatey.templates import TemplateConfig

ALLOWABLE_HTML_CONTENT_TAGS: ContextVar[set[str]] = ContextVar(
    'ALLOWABLE_HTML_CONTENT_TAGS', default={  # noqa: B039
        'address', 'aside', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'hgroup',
        'section', 'blockquote', 'dd', 'div', 'dl', 'dt', 'figcaption',
        'figure', 'hr', 'li', 'menu', 'ol', 'p', 'pre', 'ul', 'a', 'abbr',
        'b', 'bdi', 'bdo', 'br', 'cite', 'data', 'dfn', 'em', 'i', 'kbd',
        'mark', 'q', 'rp', 'rt', 'ruby', 's', 'samp', 'small', 'span',
        'strong', 'sub', 'sup', 'time', 'u', 'var', 'wbr', 'area', 'audio',
        'img', 'map', 'track', 'video', 'embed', 'object', 'picture', 'source',
        'svg', 'math', 'del', 'ins', 'caption', 'col', 'colgroup', 'table',
        'tbody', 'td', 'tfoot', 'th', 'thead', 'tr', 'button', 'details',
        'dialog', 'summary'})


def noop_escaper(value: str) -> str:
    return value


def noop_verifier(value: str) -> Literal[True]:
    return True


def html_escaper(value: str) -> str:
    return html_escape(value, quote=True)


def html_verifier(value: str) -> Literal[True]:
    parser = _HtmlVerifierParser()
    parser.feed(value)
    parser.close()
    return True


class _HtmlVerifierParser(HTMLParser):

    def handle_starttag(self, tag, attrs):
        allowlist = ALLOWABLE_HTML_CONTENT_TAGS.get()
        if tag not in allowlist:
            raise BlockedContentValue(
                'Tag not allowed in HTML content using the current allowlist',
                tag)

    def handle_endtag(self, tag):
        allowlist = ALLOWABLE_HTML_CONTENT_TAGS.get()
        if tag not in allowlist:
            raise BlockedContentValue(
                'Tag not allowed in HTML content using the current allowlist',
                tag)


# I was gonna write XML verifiers/escapers, but then I read that there are
# a bunch of DoS vulnerabilities in the stdlib XML stuff:
# https://docs.python.org/3/library/xml.html#xml-vulnerabilities
# That implies that prebaked should be a subpackage, and each submodule should
# have optional 3rd-party deps
# def xml_escaper(value: str) -> str:
#     return xml.sax.saxutils.escape(value)


html: Annotated[
    TemplateConfig,
    ClcNote(
        '''This prebaked template config uses curly braces as the interpolator
        along with a dedicated HTML escaper and verifier, with specific
        allowlisted HTML tags for context.

        Use this if you need to write HTML templates that don't inline
        javascript, CSS, etc.
        ''')
] = TemplateConfig(
    interpolator=NamedInterpolator.CURLY_BRACES,
    variable_escaper=html_escaper,
    content_verifier=html_verifier)


html_unicon: Annotated[
    TemplateConfig,
    ClcNote(
        '''This prebaked template config uses unicode control characters as the
        interpolator along with a dedicated HTML escaper and verifier, with
        specific allowlisted HTML tags for context.

        Use this if you need to write HTML templates that make use of curly
        braces within the literal template definition -- for example, if
        your template text contains inline javascript, CSS, etc.
        ''')
] = TemplateConfig(
    interpolator=NamedInterpolator.UNICODE_CONTROL,
    variable_escaper=html_escaper,
    content_verifier=html_verifier)


trusted: Annotated[
    TemplateConfig,
    ClcNote(
        '''This prebaked template config uses curly brackets as the
        interpolator, but includes **no escaping or verification**.

        Use this:
        ++  if, and **only if**, you trust all variables and content passed
            to the template
        ++  if you don't need to use curly braces within the template itself

        One example use case would be using a template for plaintext.
        ''')
] = TemplateConfig(
    interpolator=NamedInterpolator.CURLY_BRACES,
    variable_escaper=noop_escaper,
    content_verifier=noop_verifier)


trusted_unicon: Annotated[
    TemplateConfig,
    ClcNote(
        '''This prebaked template config uses unicode control characters as the
        interpolator, but includes **no escaping or verification**.

        Use this:
        ++  if, and **only if**, you trust all variables and content passed
            to the template
        ++  if you need to use curly braces within the template itself

        One example use case would be using a template as a CSS preprocessor.
        ''')
] = TemplateConfig(
    interpolator=NamedInterpolator.UNICODE_CONTROL,
    variable_escaper=noop_escaper,
    content_verifier=noop_verifier)
