from templatey.environments import AsyncTemplateLoader
from templatey.environments import SyncTemplateLoader


class DictTemplateLoader[L: object](AsyncTemplateLoader, SyncTemplateLoader):
    """A barebones template loader that simply loads templates from a
    dictionary based on whatever key you supply.
    """
    _lookup: dict[L, str]

    def __init__(self, templates: dict[L, str] | None = None):
        if templates is None:
            templates = {}
        self._lookup = templates

    def load_sync(self, template_resource_locator: L) -> str:
        return self._lookup[template_resource_locator]

    async def load_async(self, template_resource_locator: L) -> str:
        return self._lookup[template_resource_locator]
