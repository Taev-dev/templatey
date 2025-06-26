from __future__ import annotations

import typing
from contextvars import ContextVar
from random import Random
from typing import ClassVar
from typing import NamedTuple
from typing import Protocol

if typing.TYPE_CHECKING:
    from _typeshed import DataclassInstance

    from templatey.templates import TemplateConfig
    from templatey.templates import TemplateSignature
else:
    DataclassInstance = object


# Technically this should be an intersection type with both the
# _TemplateIntersectable from templates and the DataclassInstance returned by
# the dataclass transform. Unfortunately, type intersections don't yet exist in
# python, so we have to resort to this (overly broad) type
TemplateParamsInstance = DataclassInstance
type TemplateClass = type[TemplateParamsInstance]
type TemplateInstanceID = int


class TemplateIntersectable(Protocol):
    """This is the actual template protocol, which we would
    like to intersect with the TemplateParamsInstance, but cannot.
    Primarily included for documentation.
    """
    _templatey_config: ClassVar[TemplateConfig]
    # Note: whatever kind of object this is, it needs to be understood by the
    # template loader defined in the template environment. It would be nice for
    # this to be a typvar, but python doesn't currently support typevars in
    # classvars
    _templatey_resource_locator: ClassVar[object]
    _templatey_signature: ClassVar[TemplateSignature]
    # Oldschool here for performance reasons; otherwise this would be a dict.
    # Field names match the field names from the params; the value is gathered
    # from the metadata value on the field.
    _templatey_prerenderers: ClassVar[NamedTuple]


# Note: we don't need cryptographically secure IDs here, so let's preserve
# entropy (might also be faster, dunno). Also note: the only reason we're
# using a contextvar here is so that we can theoretically replace it with
# a deterministic seed during testing (if we run into flakiness due to
# non-determinism)
_ID_PRNG: ContextVar[Random] = ContextVar('_ID_PRNG', default=Random())  # noqa: B039, S311
_ID_BITS = 128


BACKSTOP_TEMPLATE_INSTANCE_ID = 0


def create_templatey_id() -> int:
    """Templatey IDs are unique identifiers (theoretically, absent
    birthday collisions) that we currently use in two places:
    ++  as a scope ID, which is used when defining templates in closures
    ++  for giving slot tree nodes a unique reference target for
        recursion loops while copying and merging, which is more robust
        than ``id(target)`` and can be transferred via dataclass field
        into cloned/copied/merged nodes.
    """
    prng = _ID_PRNG.get()
    return prng.getrandbits(_ID_BITS)
