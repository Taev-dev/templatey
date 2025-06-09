"""Does some playtesty end-to-end API tests. Yes, ideally these would be
in a different subfolder of tests, but I'm worried I'll forget to copy
them over when this moves to a dedicated repo.
"""
from __future__ import annotations

import pytest

from templatey.environments import RenderEnvironment
from templatey.interpolators import NamedInterpolator
from templatey.prebaked.env_funcs import inject_templates
from templatey.prebaked.loaders import DictTemplateLoader
from templatey.prebaked.template_configs import html
from templatey.prebaked.template_configs import html_escaper
from templatey.prebaked.template_configs import html_verifier
from templatey.templates import Content
from templatey.templates import Slot
from templatey.templates import TemplateConfig
from templatey.templates import Var
from templatey.templates import param
from templatey.templates import template


class TestApiE2E:

    def test_playtest_1(self):
        """End-to-end rendering must match expected output for scenario:
        ++  custom template interface
        ++  custom template function
        ++  template has content
        ++  template has vars
        ++  template function call references content
        ++  template function call references explicit string
        ++  awkward whitespace within the template interpolations
        """
        def href(val: str) -> tuple[str, ...]:
            return (val,)

        test_html_config = TemplateConfig(
            interpolator=NamedInterpolator.CURLY_BRACES,
            variable_escaper=html_escaper,
            content_verifier=html_verifier)

        nav = '''
            <li>
                <a href="{@href(content.target)}" class="{var.classes}">{
                    content.name}</a>
                <a href="{@href('/foo')}" class="{var.classes}">{
                    content.name}</a>
            </li>
            '''

        @template(test_html_config, 'test_template')
        class TestTemplate:
            target: Content[str]
            name: Content[str]
            classes: Var[str]

        render_env = RenderEnvironment(
            env_functions=(href,),
            template_loader=DictTemplateLoader(
                templates={'test_template': nav}))
        render_env.load_sync(TestTemplate)
        render_result = render_env.render_sync(
            TestTemplate(
                target='/some_path',
                name='Some link name',
                classes='form,morph'))

        assert render_result == '''
            <li>
                <a href="/some_path" class="form,morph">Some link name</a>
                <a href="/foo" class="form,morph">Some link name</a>
            </li>
            '''

    def test_playtest_2(self):
        def href(val: str) -> tuple[str, ...]:
            return (val,)

        nav = '''
            <li>
                <a href="{@href(content.target)}" class="{var.classes}">{
                    content.name}</a>
            </li>'''

        page = '''
            <html>
            <head><title>{content.title}</title>
            <body>
            <header>
            </header>

            <nav>
                <ol>
                {slot.nav: classes='navbar'}
                </ol>
            </nav>

            <h1>Dear {var.name}</h1>
            <main>
                {content.main}
            </main>
            <footer>
            </footer>
            </body>
            </html>
            '''

        @template(html, 'nav')
        class NavTemplate:
            target: Content[str]
            name: Content[str]
            classes: Var[str]

        @template(html, 'page')
        class PageTemplate:
            name: Var[str]
            nav: Slot[NavTemplate]
            title: Content[str]
            main: Content[str]

        render_env = RenderEnvironment(
            env_functions=(href,),
            template_loader=DictTemplateLoader(
                templates={
                    'page': page,
                    'nav': nav}))

        render_result = render_env.render_sync(
            PageTemplate(
                name='John Doe',
                title='An example page',
                main='With some content',
                nav=[
                    NavTemplate(
                        target='/',
                        name='Home',
                        classes=...),
                    NavTemplate(
                        target='/about',
                        name='About us',
                        classes=...)]))

        assert render_result == '''
            <html>
            <head><title>An example page</title>
            <body>
            <header>
            </header>

            <nav>
                <ol>
                
            <li>
                <a href="/" class="navbar">Home</a>
            </li>
            <li>
                <a href="/about" class="navbar">About us</a>
            </li>
                </ol>
            </nav>

            <h1>Dear John Doe</h1>
            <main>
                With some content
            </main>
            <footer>
            </footer>
            </body>
            </html>
            '''  # noqa: W293

    @pytest.mark.anyio
    async def test_playtest_2_async(self):
        async def href(val: str) -> tuple[str, ...]:
            return (val,)

        nav = '''
            <li>
                <a href="{@href(content.target)}" class="{var.classes}">{
                    content.name}</a>
            </li>'''

        page = '''
            <html>
            <head><title>{content.title}</title>
            <body>
            <header>
            </header>

            <nav>
                <ol>
                {slot.nav: classes='navbar'}
                </ol>
            </nav>

            <h1>Dear {var.name}</h1>
            <main>
                {content.main}
            </main>
            <footer>
            </footer>
            </body>
            </html>
            '''

        @template(html, 'nav')
        class NavTemplate:
            target: Content[str]
            name: Content[str]
            classes: Var[str]

        @template(html, 'page')
        class PageTemplate:
            name: Var[str]
            nav: Slot[NavTemplate]
            title: Content[str]
            main: Content[str]

        render_env = RenderEnvironment(
            env_functions=(href,),
            template_loader=DictTemplateLoader(
                templates={
                    'page': page,
                    'nav': nav}))

        render_result = await render_env.render_async(
            PageTemplate(
                name='John Doe',
                title='An example page',
                main='With some content',
                nav=[
                    NavTemplate(
                        target='/',
                        name='Home',
                        classes=...),
                    NavTemplate(
                        target='/about',
                        name='About us',
                        classes=...)]))

        assert render_result == '''
            <html>
            <head><title>An example page</title>
            <body>
            <header>
            </header>

            <nav>
                <ol>
                
            <li>
                <a href="/" class="navbar">Home</a>
            </li>
            <li>
                <a href="/about" class="navbar">About us</a>
            </li>
                </ol>
            </nav>

            <h1>Dear John Doe</h1>
            <main>
                With some content
            </main>
            <footer>
            </footer>
            </body>
            </html>
            '''  # noqa: W293

    def test_playtest_injection(self):
        """Injecting dynamic templates into a parent must work without
        error and produce the expected result.
        """
        def href(val: str) -> tuple[str, ...]:
            return (val,)

        nav = '''
            <li>
                <a href="{@href(content.target)}" class="{var.classes}">{
                    content.name}</a>
            </li>'''

        page = '''
            <html>
            <head><title>{content.title}</title>
            <body>
            <header>
            </header>

            <nav>
                <ol>
                {@inject_templates(content.nav)}
                </ol>
            </nav>

            <h1>Dear {var.name}</h1>
            <main>
                {content.main}
            </main>
            <footer>
            </footer>
            </body>
            </html>
            '''

        @template(html, 'nav')
        class NavTemplate:
            target: Content[str]
            name: Content[str]
            classes: Var[str]

        @template(html, 'page')
        class PageTemplate:
            name: Var[str]
            nav: Content[NavTemplate]
            title: Content[str]
            main: Content[str]

        render_env = RenderEnvironment(
            env_functions=(href, inject_templates),
            template_loader=DictTemplateLoader(
                templates={
                    'page': page,
                    'nav': nav}))

        render_result = render_env.render_sync(
            PageTemplate(
                name='John Doe',
                title='An example page',
                main='With some content',
                nav=NavTemplate(
                        target='/',
                        name='Home',
                        classes='navbar')))

        assert render_result == '''
            <html>
            <head><title>An example page</title>
            <body>
            <header>
            </header>

            <nav>
                <ol>
                
            <li>
                <a href="/" class="navbar">Home</a>
            </li>
                </ol>
            </nav>

            <h1>Dear John Doe</h1>
            <main>
                With some content
            </main>
            <footer>
            </footer>
            </body>
            </html>
            '''  # noqa: W293

    def test_with_union(self):
        """Basically the same as the second playtest, but this time with
        a union of two different navigation classes.
        """
        nav1 = '''
            <li>
                <a href="foo.html" class="{var.classes}">{
                    content.name}</a>
            </li>'''
        nav2 = '''
            <li>
                <a href="bar.html" class="{var.classes}">{
                    content.name}</a>
            </li>'''

        page = '''
            <html>
            <head><title>{content.title}</title>
            <body>
            <header>
            </header>

            <nav>
                <ol>
                {slot.nav: classes='navbar'}
                </ol>
            </nav>

            <h1>Dear {var.name}</h1>
            <main>
                {content.main}
            </main>
            <footer>
            </footer>
            </body>
            </html>
            '''

        @template(html, 'nav1')
        class NavTemplate1:
            name: Content[str]
            classes: Var[str]

        @template(html, 'nav2')
        class NavTemplate2:
            name: Content[str]
            classes: Var[str]

        @template(html, 'page')
        class PageTemplate:
            name: Var[str]
            nav: Slot[NavTemplate1 | NavTemplate2]
            title: Content[str]
            main: Content[str]

        render_env = RenderEnvironment(
            env_functions=(),
            template_loader=DictTemplateLoader(
                templates={
                    'page': page,
                    'nav1': nav1,
                    'nav2': nav2}))

        render_result = render_env.render_sync(
            PageTemplate(
                name='John Doe',
                title='An example page',
                main='With some content',
                nav=[
                    NavTemplate1(
                        name='Home',
                        classes=...),
                    NavTemplate2(
                        name='About us',
                        classes=...)]))

        assert render_result == '''
            <html>
            <head><title>An example page</title>
            <body>
            <header>
            </header>

            <nav>
                <ol>
                
            <li>
                <a href="foo.html" class="navbar">Home</a>
            </li>
            <li>
                <a href="bar.html" class="navbar">About us</a>
            </li>
                </ol>
            </nav>

            <h1>Dear John Doe</h1>
            <main>
                With some content
            </main>
            <footer>
            </footer>
            </body>
            </html>
            '''  # noqa: W293

    def test_starexp_funcs(self):
        """Star expansion must work, even when passed references.
        """
        def func_with_starrings(
                *args: str,
                **kwargs: str
                ) -> tuple[str, ...]:
            return (
                *(str(arg) for arg in args),
                *kwargs,
                *kwargs.values())

        nav = '''
            {@func_with_starrings(
                content.single_string,
                *content.multistring,
                **var.dict_)}
            '''

        @template(html, 'test_template')
        class TestTemplate:
            single_string: Content[str]
            multistring: Content[list[str]]
            dict_: Var[dict[str, str]]

        render_env = RenderEnvironment(
            env_functions=(func_with_starrings,),
            template_loader=DictTemplateLoader(
                templates={'test_template': nav}))
        render_env.load_sync(TestTemplate)
        render_result = render_env.render_sync(
            TestTemplate(
                single_string='foo',
                multistring=['oof', 'bar', 'rab'],
                # Note: this also verifies escaping is working even in a
                # recursive context
                dict_={'baz': 'zab', 'html': '<p>'}))

        assert render_result == '''
            foooofbarrabbazhtmlzab&lt;p&gt;
            '''

    def test_interp_config(self):
        """Interpolation config must handle affixes and format specs
        correctly.

        This is covering the following scenarios:
        ++  content has configured prefix, suffix, and fmt, and a value
            is passed; all must be included.
        ++  content has configured prefix, suffix, and fmt, but no
            value is passed; all must be omitted.
        ++  slot has configured prefix and suffix, and values are
            passed; all must be included for each slot instance
        ++  slot has configured prefix and suffix, but empty tuple is
            passed; all must be omitted
        """
        slot_text = '{var.value}'

        @template(html, 'slot_template')
        class SlotTemplate:
            value: Var[str]

        template_text = (
            r'''{content.omitted:
                __prefix__='^^',
                __suffix__='$$',
                __fmt__='~<5'
            }{content.configged:
                __prefix__="__",
                __suffix__=";\n",
                __fmt__='.<5'}{
            slot.nested_1: __suffix__=';\n'}{
            slot.nested_2: __prefix__='!!!!'}''')

        @template(html, 'test_template')
        class OuterTemplate:
            configged: Content[str | None]
            omitted: Content[str | None]
            nested_1: Slot[SlotTemplate]
            nested_2: Slot[SlotTemplate]

        render_env = RenderEnvironment(
            env_functions=(),
            template_loader=DictTemplateLoader(
                templates={
                    'test_template': template_text,
                    'slot_template': slot_text}))
        render_env.load_sync(OuterTemplate)
        render_result = render_env.render_sync(
            OuterTemplate(
                configged='foo',
                omitted=None,
                nested_1=(
                    SlotTemplate(value='bar'),
                    SlotTemplate(value='rab')),
                nested_2=()))

        assert render_result == '__foo..;\nbar;\nrab;\n'

    def test_renderer(self):
        """Specifying a renderer on a field must correctly alter the
        parameter value. Additionally, the ordering with respect to
        escapers and verifiers must also hold true (ie, the renderer
        runs first, then the escaper / verifier).
        """
        template_text = '''{
            content.good_content}{
            var.borderline_var}{
            var.omitted_var_value}{
            content.illegal_content_tag}'''

        @template(html, 'test_template')
        class RendererTemplate:
            good_content: Content[bool] = param(
                prerenderer=
                    lambda value: '<p>yes</p>' if value else '<p>no</p>')
            borderline_var: Var[bool] = param(
                prerenderer=lambda value: '<yes>' if value else '<no>')
            omitted_var_value: Var[str] = param(
                prerenderer=lambda value: None)
            illegal_content_tag: Content[str] = param(
                prerenderer=lambda value: 'caught!')

        render_env = RenderEnvironment(
            env_functions=(),
            template_loader=DictTemplateLoader(
                templates={'test_template': template_text,}))
        render_env.load_sync(RendererTemplate)
        render_result = render_env.render_sync(
            RendererTemplate(
                good_content=True,
                borderline_var=False,
                omitted_var_value='better not find me!',
                illegal_content_tag='<script></script>'))

        assert render_result == '<p>yes</p>&lt;no&gt;caught!'
