"""Does some playtesty end-to-end API tests. Yes, ideally these would be
in a different subfolder of tests, but I'm worried I'll forget to copy
them over when this moves to a dedicated repo.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from templatey import Content
from templatey import DynamicClassSlot
from templatey import Slot
from templatey import Var
from templatey import param
from templatey import template
from templatey.environments import RenderEnvironment
from templatey.interpolators import NamedInterpolator
from templatey.prebaked.env_funcs import inject_templates
from templatey.prebaked.loaders import DictTemplateLoader
from templatey.prebaked.template_configs import html
from templatey.prebaked.template_configs import html_escaper
from templatey.prebaked.template_configs import html_verifier
from templatey.templates import TemplateConfig


@dataclass
class VarWithFormatting:
    atom: str
    repetitions: int

    def __format__(self, format_spec: str) -> str:
        return self.atom * self.repetitions


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

    def test_forward_reference_loop_with_formatted_var(self):
        """Forward reference loops must render correctly. Variables
        with __format__ declared must correctly use their rendered value
        instead of a repr, str, or other such shenanigans. Dynamic
        class slots must work, even after injection.
        """
        def add_class():
            return ('mystyleclass',)

        # Lol where have I seen this before?
        # <div><div><div>...</div></div></div>
        div = r'''<div class="{@add_class()}">{
            slot.div:
                __prefix__="\n                ",
                __suffix__='\n                '
            }{var.body}</div>'''

        nav_section = r'''<ul>{
            slot.nav_items:
                __prefix__="\n            ",
                __suffix__="\n            "
            }</ul>'''
        nav_item = r'''<li>{
            slot.nav_item_content:
                __prefix__="\n            ",
                __suffix__="\n            "
            }</li>'''
        nav_link = r'''<a href="{content.target}" class="{@add_class()}">{
            var.name}</a>'''

        page = '''
            <html>
            <head><title>{content.title}</title>
            <body>
            <header>
            </header>

            <nav>
                {slot.nav}
            </nav>

            <h1>Dear {var.name}</h1>
            <main>
                {slot.main}
            </main>
            <footer>
                {slot.footer}
            </footer>
            </body>
            </html>
            '''

        injector = '{@inject_templates(content.to_inject)}'
        spantext = '<span>{slot.span}</span>'
        spantext_em = '<em>{var.text}</em>'
        spantext_strong = '<strong>{var.text}</strong>'

        # Note: keep this as a forward ref until we have more test coverage in
        # test_templates; this balances out the multiples_recursion test we
        # have there, which checks only the concrete case
        @template(html, 'page')
        class PageTemplate:
            name: Var[str]
            nav: Slot[NavSectionTemplate]
            title: Content[str]
            main: Slot[DivTemplate]
            footer: Slot[TemplateWithInjection]

        @template(html, 'nav_section')
        class NavSectionTemplate:
            nav_items: Slot[NavItemTemplate]

        @template(html, 'nav_item')
        class NavItemTemplate:
            nav_item_content: Slot[NavSectionTemplate | NavLinkTemplate]

        @template(html, 'nav_link')
        class NavLinkTemplate:
            target: Content[str]
            name: Var[str]

        @template(html, 'div')
        class DivTemplate:
            div: Slot[DivTemplate]
            body: Var[VarWithFormatting | None] = None

        @template(html, 'injector')
        class TemplateWithInjection:
            to_inject: Content[TextSpanTemplate]

        @template(html, 'spantext')
        class TextSpanTemplate:
            span: DynamicClassSlot

        @template(html, 'spantext_em')
        class EmTextTemplate:
            text: Var[str]

        @template(html, 'spantext_strong')
        class StrongTextTemplate:
            text: Var[str]

        render_env = RenderEnvironment(
            env_functions=(add_class, inject_templates),
            template_loader=DictTemplateLoader(
                templates={
                    'div': div,
                    'page': page,
                    'nav_section': nav_section,
                    'nav_item': nav_item,
                    'nav_link': nav_link,
                    'injector': injector,
                    'spantext': spantext,
                    'spantext_em': spantext_em,
                    'spantext_strong': spantext_strong,}))

        render_result = render_env.render_sync(
            PageTemplate(
                name='John Doe',
                title='An example page',
                main=[
                    DivTemplate(
                        div=[
                            DivTemplate(
                                div=[DivTemplate(
                                    div=(),
                                    body=VarWithFormatting(
                                        'Mainline', repetitions=2))]),
                            DivTemplate(
                                div=[DivTemplate(
                                    div=(),
                                    body=VarWithFormatting(
                                        'Sideline', repetitions=3))])])],
                nav=[
                    NavSectionTemplate(
                        nav_items=[
                            NavItemTemplate(
                                nav_item_content=[
                                    NavSectionTemplate(
                                        nav_items=[
                                            NavItemTemplate(
                                                nav_item_content=[
                                                    NavLinkTemplate(
                                                        target='/',
                                                        name='Home'),
                                                    NavLinkTemplate(
                                                        target='/blog',
                                                        name='Blog')])])])]),
                    NavSectionTemplate(
                        nav_items=[
                            NavItemTemplate(
                                nav_item_content=[
                                    NavLinkTemplate(
                                        target='/docs',
                                        name='Docs home'),
                                    NavSectionTemplate(
                                        nav_items=[
                                            NavItemTemplate(
                                                nav_item_content=[
                                                    NavLinkTemplate(
                                                        target='/docs/foo',
                                                        name='Foo docs'),
                                                    NavLinkTemplate(
                                                        target='/docs/bar',
                                                        name='Bar docs'),
                                                ])])])])],
                footer=[
                    TemplateWithInjection(
                        to_inject=TextSpanTemplate(
                            span=[
                                EmTextTemplate(text='hello, world. '),
                                StrongTextTemplate(
                                    text='is anybody listening?')]))]))

        assert render_result == '''
            <html>
            <head><title>An example page</title>
            <body>
            <header>
            </header>

            <nav>
                <ul>
            <li>
            <ul>
            <li>
            <a href="/" class="mystyleclass">Home</a>
            
            <a href="/blog" class="mystyleclass">Blog</a>
            </li>
            </ul>
            </li>
            </ul><ul>
            <li>
            <a href="/docs" class="mystyleclass">Docs home</a>
            
            <ul>
            <li>
            <a href="/docs/foo" class="mystyleclass">Foo docs</a>
            
            <a href="/docs/bar" class="mystyleclass">Bar docs</a>
            </li>
            </ul>
            </li>
            </ul>
            </nav>

            <h1>Dear John Doe</h1>
            <main>
                <div class="mystyleclass">
                <div class="mystyleclass">
                <div class="mystyleclass">MainlineMainline</div>
                </div>
                
                <div class="mystyleclass">
                <div class="mystyleclass">SidelineSidelineSideline</div>
                </div>
                </div>
            </main>
            <footer>
                <span><em>hello, world. </em><strong>is anybody listening?</strong></span>
            </footer>
            </body>
            </html>
            '''  # noqa: E501, W293

    def test_sibling_slots(self):
        """A template with two identical union slot types under
        different slot names must successfully render.
        """
        nested = '''foo{var.value}'''
        enclosing = '''{slot.foo1: value="1"} {slot.foo2: value="2"}'''

        @template(html, 'enclosing')
        class EnclosingTemplate:
            foo1: Slot[NestedTemplate | OtherNestedTemplate]
            foo2: Slot[NestedTemplate | OtherNestedTemplate]

        @template(html, 'nested')
        class NestedTemplate:
            value: Var[str]

        @template(html, 'nested')
        class OtherNestedTemplate:
            value: Var[str]

        render_env = RenderEnvironment(
            env_functions=(),
            template_loader=DictTemplateLoader(
                templates={
                    'nested': nested,
                    'enclosing': enclosing}))

        render_result = render_env.render_sync(
            EnclosingTemplate(
                foo1=[NestedTemplate(value=...)],
                foo2=[NestedTemplate(value=...)]))

        assert render_result == 'foo1 foo2'

    def test_func_with_data_ref(self):
        """A template with two identical union slot types under
        different slot names must successfully render.
        """
        template_txt = '''yolo {@echo_chamber(data.bar)}'''

        @template(html, 'template_txt')
        class FakeTemplate:
            bar: str

        def echo_chamber(val: str) -> list[str]:
            return [val]

        render_env = RenderEnvironment(
            env_functions=(echo_chamber,),
            template_loader=DictTemplateLoader(
                templates={
                    'template_txt': template_txt,}))

        render_result = render_env.render_sync(FakeTemplate(bar='oloy'))

        assert render_result == 'yolo oloy'
