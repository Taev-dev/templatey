from templatey.environments import RenderEnvironment
from templatey.prebaked.loaders import DictTemplateLoader
from templatey.prebaked.template_configs import html
from templatey.templates import Slot
from templatey.templates import Var
from templatey.templates import template

_ITERATION_COUNT = 10000000


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


def run_render_profile_simple(render_env: RenderEnvironment):
    for __ in range(_ITERATION_COUNT):
        template = NavItem(**NAV_ITEM_VARS)
        render_env.render_sync(template)


def run_render_profile_nested(render_env: RenderEnvironment):
        navigation = tuple(
            NavItem(**NAV_ITEM_VARS)
            for __ in range(PAGE_WITH_NAV_NESTED_INSTANCE_COUNT))
        for __ in range(_ITERATION_COUNT):
            template = PageWithNav(**PAGE_WITH_NAV_VARS, navigation=navigation)
            render_env.render_sync(template)


if __name__ == '__main__':
    loader = DictTemplateLoader(templates={
        'page_with_nav': TEMPLATEY_PAGE_WITH_NAV,
        'nav': TEMPLATEY_NAV_ITEM})
    render_env = RenderEnvironment(template_loader=loader)

    run_render_profile_simple(render_env)
    run_render_profile_nested(render_env)
