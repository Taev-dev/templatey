from __future__ import annotations

import typing
from collections.abc import Sequence
from contextvars import ContextVar
from random import Random
from types import EllipsisType
from typing import Annotated
from typing import ClassVar
from typing import NamedTuple
from typing import Protocol

from docnote import ClcNote
from typing_extensions import TypeIs

from templatey._annotations import InterfaceAnnotation
from templatey._annotations import InterfaceAnnotationFlavor

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


# Technically, these should use the TemplateIntersectable from templates.py
# instead of ``TemplateParamsInstance``, python doesn't support type
# intersections yet, so we settle for this.
type Slot[T: TemplateParamsInstance] = Annotated[
    Sequence[T],
    InterfaceAnnotation(InterfaceAnnotationFlavor.SLOT),
    ClcNote(
        '''A ``Slot`` is a generic container used to define nested
        templates. They must be passed a concrete template class as a
        parameter, for example:

        > A simple template with a single slot
        __embed__: 'code/python'
            @template(my_template_config, my_template_locator)
            class EnclosingTemplate:
                my_slot: Slot[MyNestedTemplate]

        If you're looking for a way to define slots using a dynamic
        template class, you should either use ``DynamicSlot``
        (preferred) or look at the prebaked template injection function.
        ''')]
type Var[T] = Annotated[
    T | EllipsisType,
    InterfaceAnnotation(InterfaceAnnotationFlavor.VARIABLE),
    ClcNote(
        '''A ``Var`` is a generic container used to define some
        interpolatable variable within a template. They should be passed
        a concrete type as a parameter, though this value is not used
        by templatey. For example:

        > A simple template with a single ``Var``
        __embed__: 'code/python'
            @template(my_template_config, my_template_locator)
            class EnclosingTemplate:
                my_var: Var[str]

        The distinction between ``Content`` and ``Var`` is in how their
        values are handled. ``Content`` is intended only for **trusted**
        content; post-interpolation values are verified, but not
        escaped. Meanwhile, ``Var`` ^^can^^ be used with untrusted
        content, and its values will be escaped, but not verified.

        Both ``Var`` escaping and ``Content`` verification are
        controlled by the ``TemplateConfig``.

        Note that ``Var`` values within templates may be provided by
        an enclosing template, as a parameter on their slots. For more
        information, <create a github issue if you read this please,
        because we haven't made a guide for this yet>
        ''')]
type Content[T] = Annotated[
    T,
    InterfaceAnnotation(InterfaceAnnotationFlavor.CONTENT),
    ClcNote(
        '''A ``Content`` parameter is a generic container used to define some
        interpolatable variable within a template. They should be passed
        a concrete type as a parameter, though this value is not used
        by templatey. For example:

        > A simple template with a single ``Content`` param
        __embed__: 'code/python'
            @template(my_template_config, my_template_locator)
            class EnclosingTemplate:
                my_var: Content[str]

        The distinction between ``Content`` and ``Var`` is in how their
        values are handled. ``Content`` is intended only for **trusted**
        content; post-interpolation values are verified, but not
        escaped. Meanwhile, ``Var`` ^^can^^ be used with untrusted
        content, and its values will be escaped, but not verified.

        Both ``Var`` escaping and ``Content`` verification are
        controlled by the ``TemplateConfig``.
        ''')]
type DynamicClassSlot[T: TemplateParamsInstance] = Annotated[
    Sequence[T],
    InterfaceAnnotation(InterfaceAnnotationFlavor.SLOT),
    InterfaceAnnotation(InterfaceAnnotationFlavor.DYNAMIC),
    ClcNote(
        '''Like a ``Slot``, a ``DynamicClassSlot`` is a generic container
        used to define nested templates. Although templates in
        templatey cannot be subclassed, they may use mixins as
        superclasses; in that case, it may be useful to supply a type
        argument to the dynamic slot to narrow its type during type
        checking.

        > A simple template with two dynamic slots
        __embed__: 'code/python'
            @template(my_template_config, my_template_locator)
            class EnclosingTemplate:
                # Note that this can be literally any template instance
                my_slot_1: DynamicClassSlot
                # This is constrained to a subclass
                my_slot_2: DynamicClassSlot[SomeMixin]

        Note that the type arg to ``DynamicClassSlot`` is neither required,
        nor used by templatey. It's provided strictly to support
        applications where you want more restrictive type annotations
        than "literally any template class whatsoever".

        Note also that ``DynamicClassSlot``s come at a slight performance
        penalty compared to normal ``Slot``s. If possible, prefer a
        plain ``Slot``, perhaps with a type union as a parameter.
        That being said, ``DynamicClassSlot``s are still more performant than
        injecting templates via environment functions.
        ''')]


def is_template_class(cls: type) -> TypeIs[type[TemplateIntersectable]]:
    """Rather than relying upon @runtime_checkable, which doesn't work
    with protocols with ClassVars, we implement our own custom checker
    here for narrowing the type against TemplateIntersectable. Note
    that this also, I think, might be usable for some of the issues
    re: the missing intersection type in python, though support might be
    unreliable depending on which type checker is in use.
    """
    return (
        hasattr(cls, '_templatey_config')
        and hasattr(cls, '_templatey_resource_locator')
        and hasattr(cls, '_templatey_signature')
    )


def is_template_instance(instance: object) -> TypeIs[TemplateIntersectable]:
    """Rather than relying upon @runtime_checkable, which doesn't work
    with protocols with ClassVars, we implement our own custom checker
    here for narrowing the type against TemplateIntersectable. Note
    that this also, I think, might be usable for some of the issues
    re: the missing intersection type in python, though support might be
    unreliable depending on which type checker is in use.
    """
    return is_template_class(type(instance))


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
