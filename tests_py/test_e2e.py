"""Does some playtesty end-to-end API tests. Yes, ideally these would be
in a different subfolder of tests, but I'm worried I'll forget to copy
them over when this moves to a dedicated repo.
"""
from __future__ import annotations

from templatey.environments import RenderEnvironment
from templatey.interpolators import NamedInterpolator
from templatey.prebaked.loaders import DictTemplateLoader
from templatey.prebaked.template_configs import html
from templatey.prebaked.template_configs import html_escaper
from templatey.prebaked.template_configs import html_verifier
from templatey.templates import Content
from templatey.templates import Slot
from templatey.templates import TemplateConfig
from templatey.templates import Var
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
            template_functions=(href,),
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
            template_functions=(href,),
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
