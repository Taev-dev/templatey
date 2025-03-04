import pytest

from templatey.exceptions import BlockedContentValue
from templatey.prebaked.template_configs import html_escaper
from templatey.prebaked.template_configs import html_verifier
from templatey.prebaked.template_configs import noop_escaper
from templatey.prebaked.template_configs import noop_verifier


class TestNoopEscaperAndVerifier:

    def test_noop_escaper(self):
        foo = 'foo'
        rv = noop_escaper(foo)
        assert rv is foo

    def test_noop_verifier(self):
        foo = 'foo'
        rv = noop_verifier(foo)
        assert rv is True


class TestHtmlEscaperAndVerifier:

    def test_html_escaper_no_tags(self):
        foo = 'foo'
        rv = html_escaper(foo)
        assert rv is foo

    def test_html_escaper_with_tags(self):
        foo = '<p>"foo"</p>'
        rv = html_escaper(foo)
        assert rv != foo
        assert '<' not in rv
        assert '>' not in rv
        assert '"' not in rv

    def test_html_verifier_with_okay_tags(self):
        foo = '<p>"foo"</p>'
        rv = html_verifier(foo)
        assert rv is True

    def test_html_verifier_with_blocked_tags(self):
        foo = '<script></script>'
        with pytest.raises(BlockedContentValue):
            html_verifier(foo)
