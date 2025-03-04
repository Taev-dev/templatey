from collections.abc import Collection

from templatey.templates import TemplateParamsInstance


def compositor[T: TemplateParamsInstance](
        template: T,
        /,
        # Again, we want this to be a narrower type, but require an
        # intersection before that's possible
        composition: dict[str, Collection[TemplateParamsInstance]]
        ) -> T:
    """The template compositor takes a toplevel template, as well as
    a collection of templates to be nested into it. This is intended
    primarily for situations where you want to describe a composition of
    multiple templates using a flat layout, rather than requiring
    multiple layers of indentation.
    """
    raise NotImplementedError
