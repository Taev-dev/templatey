from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from collections.abc import MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from dataclasses import field
from typing import ClassVar
from typing import Protocol
from typing import cast

from typing_extensions import TypeIs

from templatey._types import TemplateClass
from templatey._types import TemplateIntersectable
from templatey._types import create_templatey_id

# This is used by ``anchor_closure_scope`` to assign a scope ID to templates.
# It's used for all templates, but will only be non-null inside a closure.
_CURRENT_SCOPE_ID: ContextVar[int | None] = ContextVar(
    '_CURRENT_SCOPE_ID', default=None)

# Defining this as a contextvar basically just for testing purposes. We want
# everything else to get the default global.
PENDING_FORWARD_REFS: ContextVar[
    dict[ForwardRefLookupKey, set[TemplateClass]]] = ContextVar(
        'PENDING_FORWARD_REFS', default=defaultdict(set))  # noqa: B039


@dataclass(frozen=True, slots=True)
class ForwardRefLookupKey:
    """We use this to find all possible ForwardRefLookupKey instances
    for a particular pending template.

    Note that, by definition, forward references can only happen in two
    situations:
    ++  within the same module or closure, by something declared later
        on during execution
    ++  because of something hidden behind a ``if typing.TYPE_CHECKING``
        block in imports
    (Any other scenario would result in import failures preventing the
    module's execution).

    Names imported behind ``TYPE_CHECKING`` blocks can **only** be
    resolved using explicit helpers in the ``@template`` decorator.
    (TODO: need to add those!). There's just no way around that one;
    by definition, it's a circular import, and the name isn't available
    at runtime. So you need an escape hatch.

    Therefore, unless passed in an explicit module name because of the
    aforementioned escape hatch, these must always happen from within
    the same module as the template itself.

    Furthermore, we make one assumption here for the purposes of
    doing as much work as possible at import time, ahead of the first
    call to render a template: that the enclosing template references
    the nested template by the nested template's proper name, and
    doesn't rename it.

    The only workaround for a renamed nested template would be to
    create a dedicated resolution function, to be called at first
    render time, that re-inspects the template's type annotations, and
    figures out exactly what type it uses at that point in time. That's
    a mess, so.... we'll punt on it.
    """
    module: str
    name: str
    scope_id: int | None


def resolve_forward_references(pending_template_cls: TemplateClass):
    """The very last thing to do before we return the class after
    template decoration is to resolve all forward references inside the
    class. To do that, we first need to construct the corresponding
    ForwardRefLookupKey and check for it in the pending forward refs
    lookup.

    If we find one, we then need to update the values there, while
    checking for and correctly handling recursion.
    """
    lookup_key = ForwardRefLookupKey(
        module=pending_template_cls.__module__,
        name=pending_template_cls.__name__,
        scope_id=extract_frame_scope_id())

    forward_ref_registry = PENDING_FORWARD_REFS.get()
    dependent_template_classes = forward_ref_registry.get(lookup_key)
    if dependent_template_classes is not None:
        for dependent_template_cls in dependent_template_classes:
            dependent_xable = cast(
                TemplateIntersectable, dependent_template_cls)
            dependent_xable._templatey_signature.resolve_forward_ref(
                lookup_key, pending_template_cls)

        del forward_ref_registry[lookup_key]


# Note: mutablemapping because otherwise chainmap complains. Even though they
# aren't actually implemented, this is a quick way of getting typing to work
@dataclass(kw_only=True, slots=True)
class ForwardRefGeneratingNamespaceLookup(MutableMapping[str, type]):
    template_module: str
    template_scope_id: int | None
    captured_refs: set[ForwardRefLookupKey] = field(default_factory=set)

    def __getitem__(self, key: str) -> type:
        forward_ref = ForwardRefLookupKey(
            module=self.template_module,
            name=key,
            scope_id=self.template_scope_id)

        class ForwardReferenceProxyClass:
            """When we return a forward reference, we want to retain all
            of the expected behavior with types -- unions via ``|``,
            etc -- and therefore, we want to return a proxy class
            instead of the forward reference itself.
            """
            REFERENCE_TARGET = forward_ref

        self.captured_refs.add(forward_ref)
        return ForwardReferenceProxyClass

    # Required for mutable mapping protocol, but not for the namespace lookup.
    def __iter__(self) -> Iterator[str]:
        raise TypeError(
            'Unsupported method call in templatey foward ref implementation.')

    # Required for mutable mapping protocol, but not for the namespace lookup.
    def __len__(self) -> int:
        raise TypeError(
            'Unsupported method call in templatey foward ref implementation.')

    # Required for mutable mapping protocol, but not for the namespace lookup.
    def __setitem__(self, key, value) -> None:
        raise TypeError(
            'Unsupported method call in templatey foward ref implementation.')

    # Required for mutable mapping protocol, but not for the namespace lookup.
    def __delitem__(self, key) -> None:
        raise TypeError(
            'Unsupported method call in templatey foward ref implementation.')


@contextmanager
def anchor_closure_scope():
    """We strongly recommend against defining templates within a
    closure, as it can cause a number of fragility issues, and just
    generally makes less sense than defining templates at the module
    level. However, if you absolutely must create a new template within
    a closure, you must use ``anchor_closure_scope`` to give the
    templates a known closure scope. Can be used either as a decorator
    or a context:

    > Decorator usage
    __embed__: 'code/python'
        @anchor_closure_scope()
        def my_func():
            # template definition goes here
            ...

    > Context manager usage
    __embed__: 'code/python'
        def my_other_func():
            with anchor_closure_scope():
                # template definition goes here
                ...
    """
    token = _CURRENT_SCOPE_ID.set(create_templatey_id())
    try:
        yield
    finally:
        _CURRENT_SCOPE_ID.reset(token)


def extract_frame_scope_id() -> int | None:
    """When templates are created from inside a closure (ex, during
    testing, where this is extremely common), and forward references
    are used, we need a way to differentiate between identically-named
    templates within different functions of the same module (or the
    toplevel of the module).

    We do this via a dedicated decorator/context manager,
    ``anchor_closure_scope``, which creates a random value and assigns
    it to the corresponding context var, and then is retrieved by this
    function for use.
    """
    return _CURRENT_SCOPE_ID.get()


class ForwardReferenceProxyClass(Protocol):
    REFERENCE_TARGET: ClassVar[ForwardRefLookupKey]


def is_forward_reference_proxy(
        obj: object
        ) -> TypeIs[type[ForwardReferenceProxyClass]]:
    return isinstance(obj, type) and hasattr(obj, 'REFERENCE_TARGET')
