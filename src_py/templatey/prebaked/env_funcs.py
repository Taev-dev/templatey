from typing import Annotated

from docnote import ClcNote

from templatey.templates import TemplateParamsInstance


def xml_attrify(
        attrs: dict[str, str],
        interstitial_space: Annotated[bool,
            ClcNote('''
                Set to ``False`` if the space between attributes should
                be omitted.
                ''')] = True,
        trailing_space: Annotated[bool,
            ClcNote('''
                Set to ``True`` if a trailing space should be included
                at the very end of the result.
                ''')] = False,
        ) -> list[str]:
    """This converts a dictionary of attribute key, value pairs into
    XML attributes.

    Note that if ``attrs`` is empty, this will always return an empty
    list. This can be useful if you want to only insert a trailing space
    if attrs was non-empty.
    """
    if not attrs:
        return []

    # Optimization for speed, though we haven't actually benchmarked it for
    # comparison
    if len(attrs) == 1:
        (key, value), = attrs.items()

        if trailing_space:
            return [key, '="', value, '" ']
        else:
            return [key, '="', value, '"']

    else:
        retval = []
        for key, value in attrs.items():
            retval.extend((key, '="', value, '"'))

            if interstitial_space:
                retval.append(' ')

        if interstitial_space and not trailing_space:
            retval.pop()
        if trailing_space and not interstitial_space:
            retval.append(' ')

        return retval


def inject[T: TemplateParamsInstance](template: T) -> tuple[T]:
    """This is a very simple function to wrap a passed template instance
    into a tuple, allowing it to be used as an environment function.
    This is useful for dynamically injecting templates into parents.
    """
    return (template,)
