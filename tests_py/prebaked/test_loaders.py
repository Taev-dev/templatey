from __future__ import annotations

from pathlib import Path

import pytest

from templatey import template
from templatey.prebaked.loaders import CompanionFileLoader
from templatey.prebaked.template_configs import html

TEST_TEMPLATE_FILENAME = '_test_loader_template.html'
TEST_TEMPLATE_PATH = Path(__file__).parent / TEST_TEMPLATE_FILENAME


@template(html, TEST_TEMPLATE_FILENAME)
class FakeTemplate:
    ...


class TestCompanionFileLoader:

    def test_sync(self):
        loader = CompanionFileLoader()
        result = loader.load_sync(FakeTemplate, TEST_TEMPLATE_FILENAME)
        assert result == TEST_TEMPLATE_PATH.read_text()

    @pytest.mark.anyio
    async def test_async(self):
        loader = CompanionFileLoader()
        result = await loader.load_async(FakeTemplate, TEST_TEMPLATE_FILENAME)
        assert result == TEST_TEMPLATE_PATH.read_text()
