import json
import time
from datetime import datetime
from datetime import timezone
from pathlib import Path

import anyio
import jinja2
import pytest

from templatey.environments import RenderEnvironment
from templatey.prebaked.loaders import DictTemplateLoader
from templatey.prebaked.template_configs import html
from templatey.templates import Slot
from templatey.templates import Var
from templatey.templates import template

_ITERATION_COUNT = 1000
_OUTFILE_NAME = 'templatey_benchmark_{timestamp}.json'
_OUTFILE_DEST = Path(__file__).parent.parent


JINJA_NAV_ITEM = (
    '<li><a href="{{link}}" class="{{classlist}}">{{name}}</a></li>')
TEMPLATEY_NAV_ITEM = (
    '<li><a href="{var.link}" class="{var.classlist}">{var.name}</a></li>')

NAV_ITEM_VARS = {
    'link': '/home',
    'classlist': 'navbar',
    'name': 'Home'}


@template(html, 'nav', slots=True)
class NavItem:
    link: Var[str]
    classlist: Var[str]
    name: Var[str]


JINJA_PAGE_WITH_NAV = r'''
<!DOCTYPE html>
<html lang="en">
<head>
    <title>{{page_title}}</title>
</head>
<body>
    <ul id="navigation">
    {% for navitem in navigation %}
        <li><a href="{{navitem.link}}" class="{{navitem.classlist}}">{{navitem.name}}</a></li>
    {% endfor %}
    </ul>

    <h1>My Webpage</h1>
    {{page_content}}
</body>
</html>
'''  # noqa: E501
TEMPLATEY_PAGE_WITH_NAV = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <title>{var.page_title}</title>
</head>
<body>
    <ul id="navigation">
    {slot.navigation}
    </ul>

    <h1>My Webpage</h1>
    {var.page_content}
</body>
</html>
'''

PAGE_WITH_NAV_VARS = {
    'page_title': 'My benchmark page',
    'page_content': 'lorem ipsum'}
PAGE_WITH_NAV_NESTED_INSTANCE_COUNT = 5


def sync_footer_jinja():
    return '<footer>Thanks, world, for listening</footer>'


def sync_footer_templatey():
    return ('<footer>Thanks, world, for listening</footer>',)


async def async_footer_jinja():
    await anyio.sleep(0)
    return '<footer>Thanks, world, for listening</footer>'


async def async_footer_templatey():
    await anyio.sleep(0)
    return ('<footer>Thanks, world, for listening</footer>',)


@template(html, 'page_with_nav', slots=True)
class PageWithNav:
    page_title: Var[str]
    page_content: Var[str]
    navigation: Slot[NavItem]


JINJA_PAGE_WITH_NAV_AND_FOOTER = r'''
<!DOCTYPE html>
<html lang="en">
<head>
    <title>{{page_title}}</title>
</head>
<body>
    <ul id="navigation">
    {% for navitem in navigation %}
        <li><a href="{{navitem.link}}" class="{{navitem.classlist}}">{{navitem.name}}</a></li>
    {% endfor %}
    </ul>

    <h1>My Webpage</h1>
    {{page_content}}
    {{footer_func()}}
</body>
</html>
'''  # noqa: E501
TEMPLATEY_PAGE_WITH_NAV_AND_FOOTER = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <title>{var.page_title}</title>
</head>
<body>
    <ul id="navigation">
    {slot.navigation}
    </ul>

    <h1>My Webpage</h1>
    {var.page_content}
    {@footer_func()}
</body>
</html>
'''


@template(html, 'page_with_nav_and_footer', slots=True)
class PageWithNavAndFooter:
    page_title: Var[str]
    page_content: Var[str]
    navigation: Slot[NavItem]


@pytest.fixture(scope='session')
def benchmark_gatherer():
    now = datetime.now(timezone.utc)
    timestamp = f"{now.strftime('%Y-%m-%d')}-{int(now.timestamp())}"
    outfile_path = _OUTFILE_DEST / _OUTFILE_NAME.format(timestamp=timestamp)
    results = {'jinja': {}, 'templatey': {}}
    try:
        yield results
    finally:
        outfile_path.write_text(json.dumps(results))


@pytest.fixture
def jinja_env() -> jinja2.Environment:
    return jinja2.Environment(autoescape=True, loader=None, enable_async=True)


@pytest.fixture
def templatey_env() -> RenderEnvironment:
    loader = DictTemplateLoader(templates={
        'page_with_nav_and_footer': TEMPLATEY_PAGE_WITH_NAV_AND_FOOTER,
        'page_with_nav': TEMPLATEY_PAGE_WITH_NAV,
        'nav': TEMPLATEY_NAV_ITEM})
    return RenderEnvironment(template_loader=loader)


@pytest.mark.benchmark
class TestBenchmarks:

    def test_simple_load_jinja(self, jinja_env, benchmark_gatherer):
        """Benchmark loading (and only loading) a very simple template
        (nav item). This evaluates template parsing speed.
        """
        elapsed_time = 0
        for __ in range(_ITERATION_COUNT):
            before = time.monotonic()
            jinja_env.from_string(JINJA_NAV_ITEM)
            after = time.monotonic()
            elapsed_time += (after - before)
        benchmark_gatherer['jinja']['simple.load'] = (
            elapsed_time / _ITERATION_COUNT)

    def test_simple_load_templatey(self, templatey_env, benchmark_gatherer):
        """Benchmark loading (and only loading) a very simple template
        (nav item). This evaluates template parsing speed.
        """
        elapsed_time = 0
        for __ in range(_ITERATION_COUNT):
            before = time.monotonic()
            templatey_env.load_sync(NavItem)
            after = time.monotonic()
            elapsed_time += (after - before)
        benchmark_gatherer['templatey']['simple.load'] = (
            elapsed_time / _ITERATION_COUNT)

    def test_simple_render_jinja(self, jinja_env, benchmark_gatherer):
        """Test rendering (and only rendering) a very simple template
        (nav item). This evaluates how quickly we can create the final
        output string from a pre-loaded template and its parameters.
        """
        elapsed_time = 0
        template = jinja_env.from_string(JINJA_NAV_ITEM)
        for __ in range(_ITERATION_COUNT):
            before = time.monotonic()
            template.render(**NAV_ITEM_VARS)
            after = time.monotonic()
            elapsed_time += (after - before)
        benchmark_gatherer['jinja']['simple.render'] = (
            elapsed_time / _ITERATION_COUNT)

    def test_simple_render_templatey(self, templatey_env, benchmark_gatherer):
        """Test rendering (and only rendering) a very simple template
        (nav item). This evaluates how quickly we can create the final
        output string from a pre-loaded template and its parameters.
        """
        elapsed_time = 0
        template = templatey_env.load_sync(NavItem)
        for __ in range(_ITERATION_COUNT):
            template = NavItem(**NAV_ITEM_VARS)
            before = time.monotonic()
            templatey_env.render_sync(template)
            after = time.monotonic()
            elapsed_time += (after - before)
        benchmark_gatherer['templatey']['simple.render'] = (
            elapsed_time / _ITERATION_COUNT)

    def test_nested_component_load_jinja(self, jinja_env, benchmark_gatherer):
        """Benchmark loading (and only loading) a slightly more complex
        template that includes nested components (page with nav).
        This evaluates template parsing speed.
        """
        elapsed_time = 0
        for __ in range(_ITERATION_COUNT):
            before = time.monotonic()
            jinja_env.from_string(JINJA_PAGE_WITH_NAV)
            after = time.monotonic()
            elapsed_time += (after - before)
        benchmark_gatherer['jinja']['nested_comp.load'] = (
            elapsed_time / _ITERATION_COUNT)

    def test_nested_component_load_templatey(
            self, templatey_env, benchmark_gatherer):
        """Benchmark loading (and only loading) a slightly more complex
        template that includes nested components (page with nav).
        This evaluates template parsing speed.
        """
        elapsed_time = 0
        for __ in range(_ITERATION_COUNT):
            before = time.monotonic()
            templatey_env.load_sync(PageWithNav)
            after = time.monotonic()
            elapsed_time += (after - before)
        benchmark_gatherer['templatey']['nested_comp.load'] = (
            elapsed_time / _ITERATION_COUNT)

    def test_nested_component_render_jinja(
            self, jinja_env, benchmark_gatherer):
        """Test rendering (and only rendering) a slightly more complex
        template that includes nested components (page with nav).
        This evaluates how quickly we can create the final
        output string from a pre-loaded template and its parameters.
        """
        elapsed_time = 0
        template = jinja_env.from_string(JINJA_PAGE_WITH_NAV)
        navigation = [
            NAV_ITEM_VARS for __ in range(PAGE_WITH_NAV_NESTED_INSTANCE_COUNT)]
        for __ in range(_ITERATION_COUNT):
            before = time.monotonic()
            template.render(**NAV_ITEM_VARS, navigation=navigation)
            after = time.monotonic()
            elapsed_time += (after - before)
        benchmark_gatherer['jinja']['nested_comp.render'] = (
            elapsed_time / _ITERATION_COUNT)

    def test_nested_component_render_templatey(
            self, templatey_env, benchmark_gatherer):
        """Test rendering (and only rendering) a slightly more complex
        template that includes nested components (page with nav).
        This evaluates how quickly we can create the final
        output string from a pre-loaded template and its parameters.
        """
        elapsed_time = 0
        templatey_env.load_sync(PageWithNav)
        templatey_env.load_sync(NavItem)
        navigation = tuple(
            NavItem(**NAV_ITEM_VARS)
            for __ in range(PAGE_WITH_NAV_NESTED_INSTANCE_COUNT))
        for __ in range(_ITERATION_COUNT):
            template = PageWithNav(**PAGE_WITH_NAV_VARS, navigation=navigation)
            before = time.monotonic()
            templatey_env.render_sync(template)
            after = time.monotonic()
            elapsed_time += (after - before)
        benchmark_gatherer['templatey']['nested_comp.render'] = (
            elapsed_time / _ITERATION_COUNT)

    def test_nested_component_with_funcs_render_jinja_sync(
            self, jinja_env, benchmark_gatherer):
        """Test rendering (and only rendering) a slightly more complex
        template that includes nested components and function calls.
        This evaluates how quickly we can create the final
        output string from a pre-loaded template and its parameters.
        """
        elapsed_time = 0
        template = jinja_env.from_string(JINJA_PAGE_WITH_NAV_AND_FOOTER)
        navigation = [
            NAV_ITEM_VARS for __ in range(PAGE_WITH_NAV_NESTED_INSTANCE_COUNT)]
        for __ in range(_ITERATION_COUNT):
            before = time.monotonic()
            template.render(
                **NAV_ITEM_VARS,
                navigation=navigation, footer_func=sync_footer_jinja)
            after = time.monotonic()
            elapsed_time += (after - before)
        benchmark_gatherer['jinja']['nested_comp_funky.render.sync'] = (
                elapsed_time / _ITERATION_COUNT)

    def test_nested_component_with_funcs_render_templatey_sync(
            self, templatey_env, benchmark_gatherer):
        """Test rendering (and only rendering) a slightly more complex
        template that includes nested components and function calls.
        This evaluates how quickly we can create the final
        output string from a pre-loaded template and its parameters.
        """
        elapsed_time = 0
        templatey_env.register_env_function(
            sync_footer_templatey, with_name='footer_func')
        templatey_env.load_sync(PageWithNavAndFooter)
        templatey_env.load_sync(PageWithNav)
        templatey_env.load_sync(NavItem)
        navigation = tuple(
            NavItem(**NAV_ITEM_VARS)
            for __ in range(PAGE_WITH_NAV_NESTED_INSTANCE_COUNT))
        for __ in range(_ITERATION_COUNT):
            template = PageWithNavAndFooter(
                **PAGE_WITH_NAV_VARS, navigation=navigation)
            before = time.monotonic()
            templatey_env.render_sync(template)
            after = time.monotonic()
            elapsed_time += (after - before)
        benchmark_gatherer['templatey']['nested_comp_funky.render.sync'] = (
                elapsed_time / _ITERATION_COUNT)

    @pytest.mark.anyio
    async def test_nested_component_with_funcs_render_jinja_async(
            self, jinja_env, benchmark_gatherer, anyio_backend_name):
        """Test rendering (and only rendering) a slightly more complex
        template that includes nested components and function calls.
        This evaluates how quickly we can create the final
        output string from a pre-loaded template and its parameters.
        """
        elapsed_time = 0
        template = jinja_env.from_string(JINJA_PAGE_WITH_NAV_AND_FOOTER)
        navigation = [
            NAV_ITEM_VARS for __ in range(PAGE_WITH_NAV_NESTED_INSTANCE_COUNT)]
        for __ in range(_ITERATION_COUNT):
            before = time.monotonic()
            await template.render_async(
                **NAV_ITEM_VARS,
                navigation=navigation, footer_func=async_footer_jinja)
            after = time.monotonic()
            elapsed_time += (after - before)
        benchmark_gatherer['jinja'][
            f'nested_comp_funky.render.{anyio_backend_name}'] = (
                elapsed_time / _ITERATION_COUNT)

    @pytest.mark.anyio
    async def test_nested_component_with_funcs_render_templatey_async(
            self, templatey_env, benchmark_gatherer, anyio_backend_name):
        """Test rendering (and only rendering) a slightly more complex
        template that includes nested components and function calls.
        This evaluates how quickly we can create the final
        output string from a pre-loaded template and its parameters.
        """
        elapsed_time = 0
        templatey_env.register_env_function(
            async_footer_templatey, with_name='footer_func')
        templatey_env.load_sync(PageWithNavAndFooter)
        templatey_env.load_sync(PageWithNav)
        templatey_env.load_sync(NavItem)
        navigation = tuple(
            NavItem(**NAV_ITEM_VARS)
            for __ in range(PAGE_WITH_NAV_NESTED_INSTANCE_COUNT))
        for __ in range(_ITERATION_COUNT):
            template = PageWithNavAndFooter(
                **PAGE_WITH_NAV_VARS, navigation=navigation)
            before = time.monotonic()
            await templatey_env.render_async(template)
            after = time.monotonic()
            elapsed_time += (after - before)
        benchmark_gatherer['templatey'][
            f'nested_comp_funky.render.{anyio_backend_name}'] = (
                elapsed_time / _ITERATION_COUNT)
