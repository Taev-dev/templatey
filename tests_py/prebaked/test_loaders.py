from __future__ import annotations

from pathlib import Path

import pytest

from templatey import Var
from templatey import template
from templatey.prebaked.loaders import CompanionFileLoader
from templatey.prebaked.template_configs import html
from templatey.prebaked.template_configs import html_unicon

THISDIR = Path(__file__).parent
TEST_TEMPLATE_FILENAME = '_test_loader_template.html'
TEST_TEMPLATE_FILENAME_UNICODE = '_test_loader_template_unicode.html'
TEST_TEMPLATE_PATH = THISDIR / TEST_TEMPLATE_FILENAME
TEST_TEMPLATE_PATH_UNICODE = THISDIR / TEST_TEMPLATE_FILENAME_UNICODE


@template(html, TEST_TEMPLATE_FILENAME)
class FakeTemplate:
    foo: Var[str]


@template(html_unicon, TEST_TEMPLATE_FILENAME)
class FakeTemplateUnicode:
    foo: Var[str]


class TestCompanionFileLoader:

    def test_sync(self):
        loader = CompanionFileLoader()
        result = loader.load_sync(FakeTemplate, TEST_TEMPLATE_FILENAME)
        assert result == TEST_TEMPLATE_PATH.read_text()

    def test_sync_unicode(self):
        """Loading must also work with unicode inside the file."""
        loader = CompanionFileLoader()
        result = loader.load_sync(
            FakeTemplateUnicode, TEST_TEMPLATE_FILENAME_UNICODE)
        assert result == TEST_TEMPLATE_PATH_UNICODE.read_text('utf-8')

    @pytest.mark.anyio
    async def test_async(self):
        loader = CompanionFileLoader()
        result = await loader.load_async(FakeTemplate, TEST_TEMPLATE_FILENAME)
        assert result == TEST_TEMPLATE_PATH.read_text()

    @pytest.mark.anyio
    async def test_async_unicode(self):
        """Loading must also work with unicode inside the file."""
        loader = CompanionFileLoader()
        result = await loader.load_async(
            FakeTemplateUnicode, TEST_TEMPLATE_FILENAME_UNICODE)
        assert result == TEST_TEMPLATE_PATH_UNICODE.read_text('utf-8')
