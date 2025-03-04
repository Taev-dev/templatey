import json
import time
from datetime import datetime
from datetime import timezone
from pathlib import Path

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


@template(html, 'nav')
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


@template(html, 'page_with_nav')
class PageWithNav:
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
    return jinja2.Environment(autoescape=True, loader=None)


@pytest.fixture
def templatey_env() -> RenderEnvironment:
    loader = DictTemplateLoader(templates={
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
