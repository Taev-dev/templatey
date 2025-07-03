from __future__ import annotations

from collections import defaultdict
from dataclasses import FrozenInstanceError
from dataclasses import is_dataclass
from typing import cast
from typing import get_type_hints
from unittest.mock import MagicMock

import pytest

from templatey._forwardrefs import PENDING_FORWARD_REFS
from templatey._types import Content
from templatey._types import DynamicClassSlot
from templatey._types import Slot
from templatey._types import TemplateIntersectable
from templatey._types import Var
from templatey._types import is_template_class
from templatey._types import is_template_instance
from templatey.templates import make_template_definition
from templatey.templates import template

from templatey_testutils import fake_template_config


class TestIsTemplateClass:

    def test_positive(self):
        class FakeTemplate:
            ...

        # Quick and dirty. This will break if we add anything that isn't a
        # class var to the TemplateIntersectable class.
        for key in get_type_hints(
            TemplateIntersectable,
            localns=defaultdict(MagicMock)
        ):
            setattr(FakeTemplate, key, object())

        assert is_template_class(FakeTemplate)

    def test_negative(self):
        class FakeTemplate:
            ...

        assert not is_template_class(FakeTemplate)


class TestIsTemplateInstance:
    def test_positive(self):
        class FakeTemplate:
            ...

        # Quick and dirty. This will break if we add anything that isn't a
        # class var to the TemplateIntersectable class.
        for key in get_type_hints(
            TemplateIntersectable,
            localns=defaultdict(MagicMock)
        ):
            setattr(FakeTemplate, key, object())

        instance = FakeTemplate()
        assert is_template_instance(instance)

    def test_negative(self):
        class FakeTemplate:
            ...
        instance = FakeTemplate()

        assert not is_template_instance(instance)


class TestTemplate:
    """template()
    """

    def test_works(self):
        """The template decorator must complete without error and
        result in a template class.
        """
        @template(fake_template_config, object())
        class FakeTemplate:
            ...

        assert isinstance(FakeTemplate, type)
        assert is_template_class(FakeTemplate)

    def test_supports_passthrough(self):
        """Additional params must be passed through to the dataclass
        decorator.
        """
        @template(fake_template_config, object(), frozen=True)
        class FakeTemplate:
            foo: Var[str]

        instance = FakeTemplate(foo='foo')
        with pytest.raises(FrozenInstanceError):
            instance.foo = 'bar'  # type: ignore


class TestMakeTemplateDefinition:
    """make_template_definition()
    """

    def test_simplest_case(self):
        """The required bookkeeping variables must be added to the
        class.
        """
        class Foo:
            foo: Var[str]

        retval = make_template_definition(
            Foo,
            dataclass_kwargs={},
            template_resource_locator=object(),
            template_config=fake_template_config)
        assert is_template_class(retval)

    def test_closure_resolution_works(self):
        """Another template referenced as a slot must successfully
        return, even if that template was defined in a closure.

        This is specifically targeting our workaround to get_type_hints
        needing access to the function locals within the closure.
        """
        @template(fake_template_config, object())
        class Foo:
            foo: Var[str]

        class Bar:
            foo: Slot[Foo]

        retval = make_template_definition(
            Bar,
            dataclass_kwargs={},
            template_resource_locator=object(),
            template_config=fake_template_config)
        assert is_template_class(retval)

    def test_forward_ref_works(self):
        """Slots must be definable using forward references, and these
        forward references must be recorded on the template alongside
        the rest of the slot tree.

        Once the reference is available, it must be resolved on the
        forward-referencing class.
        """
        class Bar:
            foo: Slot[Foo]

        retval = cast(type[TemplateIntersectable], make_template_definition(
            Bar,
            dataclass_kwargs={},
            template_resource_locator=object(),
            template_config=fake_template_config))
        assert is_template_class(retval)

        assert len(retval._templatey_signature._pending_ref_lookup) == 1
        pending_ref = next(iter(
            retval._templatey_signature._pending_ref_lookup))
        assert pending_ref.name == 'Foo'

        forward_ref_registry = PENDING_FORWARD_REFS.get()
        assert len(forward_ref_registry) == 1
        assert pending_ref in forward_ref_registry
        assert forward_ref_registry[pending_ref] == {Bar}

        @template(fake_template_config, object())
        class Foo:
            foo: Var[str]

        assert len(retval._templatey_signature._pending_ref_lookup) == 0
        assert not forward_ref_registry
        assert Foo in retval._templatey_signature._slot_tree_lookup

    def test_double_forward_ref(self):
        """Slots must be definable using forward references, and these
        forward references must be recorded on the template alongside
        the rest of the slot tree.

        Once the reference is available, it must be resolved on the
        forward-referencing class.

        This expands on the plain forward ref test case by having
        additional forward and backward refs on the resolved template
        class.
        """
        class Bar:
            foo: Slot[Foo]

        retval = cast(type[TemplateIntersectable], make_template_definition(
            Bar,
            dataclass_kwargs={},
            template_resource_locator=object(),
            template_config=fake_template_config))
        assert is_template_class(retval)

        assert len(retval._templatey_signature._pending_ref_lookup) == 1
        pending_ref = next(iter(
            retval._templatey_signature._pending_ref_lookup))
        assert pending_ref.name == 'Foo'

        forward_ref_registry = PENDING_FORWARD_REFS.get()
        assert len(forward_ref_registry) == 1
        assert pending_ref in forward_ref_registry
        assert forward_ref_registry[pending_ref] == {Bar}

        @template(fake_template_config, object())
        class Baz:
            zab: Var[str]

        @template(fake_template_config, object())
        class Foo:
            oof: Var[str]
            baz: Slot[Baz]
            runout: Slot[IRanOutOfFooLikeNames]

        assert len(retval._templatey_signature._pending_ref_lookup) == 1
        pending_ref = next(iter(
            retval._templatey_signature._pending_ref_lookup))
        assert pending_ref.name == 'IRanOutOfFooLikeNames'

        assert len(forward_ref_registry) == 1
        assert pending_ref in forward_ref_registry
        assert forward_ref_registry[pending_ref] == {Bar, Foo}

        assert Foo in retval._templatey_signature._slot_tree_lookup
        assert Baz in retval._templatey_signature._slot_tree_lookup

        @template(fake_template_config, object())
        class IRanOutOfFooLikeNames:
            ranout: Var[str]

        assert len(retval._templatey_signature._pending_ref_lookup) == 0
        assert not forward_ref_registry
        assert (
            IRanOutOfFooLikeNames
            in retval._templatey_signature._slot_tree_lookup)

    def test_nested_forward_ref_works(self):
        """Slots must be definable using forward references, and these
        forward references must be recorded on the template alongside
        the rest of the slot tree.

        Once the reference is available, it must be resolved on the
        forward-referencing class.

        This expands on the plain forward ref test case by making sure
        that forward references are correctly passed along from nested
        templates to their enclosing templates.
        """
        @template(fake_template_config, object())
        class Bar:
            foo: Slot[Foo]

        class Baz:
            bar: Slot[Bar]

        retval = cast(type[TemplateIntersectable], make_template_definition(
            Baz,
            dataclass_kwargs={},
            template_resource_locator=object(),
            template_config=fake_template_config))
        assert is_template_class(retval)

        assert len(retval._templatey_signature._pending_ref_lookup) == 1
        pending_ref = next(iter(
            retval._templatey_signature._pending_ref_lookup))
        assert pending_ref.name == 'Foo'

        forward_ref_registry = PENDING_FORWARD_REFS.get()
        assert len(forward_ref_registry) == 1
        assert pending_ref in forward_ref_registry
        assert forward_ref_registry[pending_ref] == {Bar, Baz}

        @template(fake_template_config, object())
        class Foo:
            foo: Var[str]

        assert len(retval._templatey_signature._pending_ref_lookup) == 0
        assert not forward_ref_registry
        assert Foo in retval._templatey_signature._slot_tree_lookup

    def test_simple_recursion_works(self):
        """Slots must support recursive references back to the current
        template, and these must not be considered pending references.
        """
        class Foo:
            # This works if you decorate the class, but not without decoration.
            # Since we're testing make_template_definition independently of
            # the decorator, the pragmatic thing is to just ignore here.
            foo: Slot[Foo]  # type: ignore

        retval = cast(type[TemplateIntersectable], make_template_definition(
            Foo,
            dataclass_kwargs={},
            template_resource_locator=object(),
            template_config=fake_template_config))
        assert is_template_class(retval)
        assert len(retval._templatey_signature._pending_ref_lookup) == 0
        forward_ref_registry = PENDING_FORWARD_REFS.get()
        assert not forward_ref_registry
        assert Foo in retval._templatey_signature._slot_tree_lookup

    def test_recursion_loop_works(self):
        """Slots must support recursive reference loops back to the
        current template using forward references. Initially, the loop
        must be considered a forward reference, but once resolved, it
        must not be pending on either class.

        Once the reference is available, it must be resolved on the
        forward-referencing class.
        """
        class Bar:
            foo: Slot[Foo]

        retval = cast(type[TemplateIntersectable], make_template_definition(
            Bar,
            dataclass_kwargs={},
            template_resource_locator=object(),
            template_config=fake_template_config))
        assert is_template_class(retval)

        assert len(retval._templatey_signature._pending_ref_lookup) == 1
        pending_ref = next(iter(
            retval._templatey_signature._pending_ref_lookup))
        assert pending_ref.name == 'Foo'

        forward_ref_registry = PENDING_FORWARD_REFS.get()
        assert len(forward_ref_registry) == 1
        assert pending_ref in forward_ref_registry
        assert forward_ref_registry[pending_ref] == {Bar}

        @template(fake_template_config, object())
        class Foo:
            # This works if you decorate the class, but not without decoration.
            # Since we're testing make_template_definition independently of
            # the decorator, the pragmatic thing is to just ignore here.
            bar: Slot[Bar]  # type: ignore
        foo_xable = cast(TemplateIntersectable, Foo)

        assert not forward_ref_registry
        assert len(retval._templatey_signature._pending_ref_lookup) == 0
        assert not forward_ref_registry
        assert Foo in foo_xable._templatey_signature._slot_tree_lookup
        assert Foo in retval._templatey_signature._slot_tree_lookup

    def test_config_and_locator_assigned(self):
        """The passed template config must be stored on the class, along
        with the template locator.
        """

        class FakeTemplate:
            bar: Var[str]

        retval = cast(type[TemplateIntersectable], make_template_definition(
            FakeTemplate,
            dataclass_kwargs={},
            template_resource_locator='my_special_locator',
            template_config=fake_template_config))
        assert retval._templatey_resource_locator == 'my_special_locator'
        assert retval._templatey_config is fake_template_config

    def test_slot_extraction(self):
        """Fields declared with Slot[...] must be correctly detected
        and stored on the class.
        """
        @template(fake_template_config, object())
        class Foo:
            foo: Var[str]

        class FakeTemplate:
            foo: Slot[Foo]
            bar: Var[str]
            baz: Content[str]

        retval = cast(type[TemplateIntersectable], make_template_definition(
            FakeTemplate,
            dataclass_kwargs={},
            template_resource_locator=object(),
            template_config=fake_template_config))
        signature = retval._templatey_signature

        assert len(signature.slot_names) == 1
        assert 'foo' in signature.slot_names

    def test_slot_extraction_with_union(self):
        """Slots declared as the union of two templates must correctly
        include both referenced child templates in the loaded parent.
        """
        @template(fake_template_config, object())
        class Foo:
            foo: Var[str]

        @template(fake_template_config, object())
        class Bar:
            bar: Var[str]

        class FakeTemplate:
            foo: Slot[Foo | Bar]
            bar: Var[str]
            baz: Content[str]

        retval = cast(type[TemplateIntersectable], make_template_definition(
            FakeTemplate,
            dataclass_kwargs={},
            template_resource_locator=object(),
            template_config=fake_template_config))
        signature = retval._templatey_signature

        assert len(signature.slot_names) == 1
        assert 'foo' in signature.slot_names

    def test_slot_multiples_union(self):
        """Templates with multiple slots of the same union type must be
        correctly defined, with both separate routes in the slot tree.
        """
        class FakeTemplate:
            foo1: Slot[Foo | Bar | Baz]
            foo2: Slot[Foo | Bar | Baz]

        @template(fake_template_config, object())
        class Foo:
            foo: Var[str]

        @template(fake_template_config, object())
        class Bar:
            foo: Var[str]

        @template(fake_template_config, object())
        class Baz:
            foo: Var[str]

        retval = cast(type[TemplateIntersectable], make_template_definition(
            FakeTemplate,
            dataclass_kwargs={},
            template_resource_locator=object(),
            template_config=fake_template_config))
        signature = retval._templatey_signature

        assert len(signature.slot_names) == 2
        assert 'foo1' in signature.slot_names
        assert 'foo2' in signature.slot_names
        root_node = signature._slot_tree_lookup[Foo]
        assert root_node.has_route_for('foo1', Foo)
        assert root_node.has_route_for('foo2', Foo)

    def test_slot_multiples_recursion(self):
        """Templates with multiple slots of the same recursive type must
        be correctly defined, with both separate routes in the slot
        tree.
        """
        @template(fake_template_config, object())
        class FakeTemplateFordref:
            slot1: Slot[Baz | Foo | Bar]
            slot2: Slot[Baz | Foo | Bar]

        @template(fake_template_config, object())
        class Foo:
            bar_or_baz: Slot[Bar | Baz]

        @template(fake_template_config, object())
        class Bar:
            foo: Slot[Foo]

        @template(fake_template_config, object())
        class Baz:
            value: Var[str]

        @template(fake_template_config, object())
        class FakeTemplateBackref:
            slot1: Slot[Baz | Foo | Bar]
            slot2: Slot[Baz | Foo | Bar]

        retval_backref = cast(type[TemplateIntersectable], FakeTemplateBackref)
        retval_fordref = cast(type[TemplateIntersectable], FakeTemplateFordref)
        signature_backref = retval_backref._templatey_signature
        signature_fordref = retval_fordref._templatey_signature

        assert signature_backref.slot_names == {'slot1', 'slot2'}
        assert signature_fordref.slot_names == {'slot1', 'slot2'}
        assert set(signature_backref._slot_tree_lookup) == {Foo, Bar, Baz}
        assert set(signature_fordref._slot_tree_lookup) == {Foo, Bar, Baz}
        assert not signature_backref._pending_ref_lookup
        assert not signature_fordref._pending_ref_lookup

        # Hard-coding the expected tree is a LOT of tedious manual work, so
        # instead we're just going to be as pragmatic as possible and just
        # test backref against forward-ref
        foo_root_backref = signature_backref._slot_tree_lookup[Foo]
        foo_root_fordref = signature_fordref._slot_tree_lookup[Foo]
        assert foo_root_backref.is_equivalent(foo_root_fordref)
        assert {slot.slot_path for slot in foo_root_backref} == {
            ('slot1', Foo),
            ('slot1', Bar),
            ('slot2', Foo),
            ('slot2', Bar),}

        bar_root_backref = signature_backref._slot_tree_lookup[Bar]
        bar_root_fordref = signature_fordref._slot_tree_lookup[Bar]
        assert bar_root_backref.is_equivalent(bar_root_fordref)
        assert {slot.slot_path for slot in bar_root_backref} == {
            ('slot1', Foo),
            ('slot1', Bar),
            ('slot2', Foo),
            ('slot2', Bar),}

        baz_root_backref = signature_backref._slot_tree_lookup[Baz]
        baz_root_fordref = signature_fordref._slot_tree_lookup[Baz]
        assert baz_root_backref.is_equivalent(baz_root_fordref)
        assert {slot.slot_path for slot in baz_root_backref} == {
            ('slot1', Foo),
            ('slot1', Bar),
            ('slot1', Baz),
            ('slot2', Baz),
            ('slot2', Foo),
            ('slot2', Bar),}

    def test_var_extraction(self):
        """Fields declared with Var[...] must be correctly detected
        and stored on the class.
        """
        class FakeTemplate:
            bar: Var[str]
            baz: Content[str]

        retval = cast(type[TemplateIntersectable], make_template_definition(
            FakeTemplate,
            dataclass_kwargs={},
            template_resource_locator=object(),
            template_config=fake_template_config))
        signature = retval._templatey_signature

        assert len(signature.var_names) == 1
        assert 'bar' in signature.var_names

    def test_content_extraction(self):
        """Fields declared with Content[...] must be correctly detected
        and stored on the class.
        """
        class FakeTemplate:
            bar: Var[str]
            baz: Content[str]

        retval = cast(type[TemplateIntersectable], make_template_definition(
            FakeTemplate,
            dataclass_kwargs={},
            template_resource_locator=object(),
            template_config=fake_template_config))
        signature = retval._templatey_signature

        assert len(signature.content_names) == 1
        assert 'baz' in signature.content_names

    def test_is_dataclass(self):
        """The template maker must also convert the class to a
        dataclass.
        """
        class FakeTemplate:
            foo: Var[str]

        retval = make_template_definition(
            FakeTemplate,
            dataclass_kwargs={},
            template_resource_locator=object(),
            template_config=fake_template_config)
        assert is_dataclass(retval)

    def test_requires_interface_typehint(self):
        """Template type definitions must only include Vars, Contents,
        and Slots, and must error if anything else is passed.
        """
        with pytest.raises(TypeError):
            class FakeTemplate:
                foo: str

            make_template_definition(
                FakeTemplate,
                dataclass_kwargs={},
                template_resource_locator=object(),
                template_config=fake_template_config)

    def test_supports_passthrough(self):
        """Dataclass kwargs must be forwarded to the dataclass
        constructor.
        """
        class FakeTemplate:
            foo: Var[str]

        template_cls = make_template_definition(
            FakeTemplate,
            dataclass_kwargs={'frozen': True, 'slots': True},
            template_resource_locator=object(),
            template_config=fake_template_config)

        instance = template_cls(foo='foo')  # type: ignore
        with pytest.raises(FrozenInstanceError):
            instance.foo = 'bar'  # type: ignore

        assert hasattr(instance, '__slots__')

    def test_dynamic_class_slot_extraction(self):
        """Fields declared with DynamicClassSlot[...] must be correctly
        detected and stored on the class.

        This checks the simplest case, with no forward references nor
        nesting slots.
        """
        @template(fake_template_config, object())
        class Foo:
            foo: Var[str]

        class FakeTemplate:
            foo: Slot[Foo]
            bar: DynamicClassSlot

        retval = cast(type[TemplateIntersectable], make_template_definition(
            FakeTemplate,
            dataclass_kwargs={},
            template_resource_locator=object(),
            template_config=fake_template_config))
        signature = retval._templatey_signature

        assert len(signature.slot_names) == 1
        assert 'foo' in signature.slot_names
        assert len(signature.dynamic_class_slot_names) == 1
        assert 'bar' in signature.dynamic_class_slot_names

        assert len(signature._dynamic_class_slot_tree) == 0
        assert signature._dynamic_class_slot_tree.dynamic_class_slot_names == {
            'bar'}

    def test_dynamic_class_slot_extraction_recursion_loop(self):
        """Fields declared with DynamicClassSlot[...] must be correctly
        detected and stored on the class.

        This checks the simplest case, with no forward references nor
        nesting slots.
        """
        @template(fake_template_config, object())
        class FakeTemplateFordref:
            slot1: Slot[Baz | Foo | Bar]
            slot2: Slot[Baz | Foo | Bar]

        @template(fake_template_config, object())
        class Foo:
            bar_or_baz: Slot[Bar | Baz]

        @template(fake_template_config, object())
        class Bar:
            foo: Slot[Foo]
            dynamico: DynamicClassSlot

        @template(fake_template_config, object())
        class Baz:
            value: Var[str]

        @template(fake_template_config, object())
        class FakeTemplateBackref:
            slot1: Slot[Baz | Foo | Bar]
            slot2: Slot[Baz | Foo | Bar]

        retval_backref = cast(type[TemplateIntersectable], FakeTemplateBackref)
        retval_fordref = cast(type[TemplateIntersectable], FakeTemplateFordref)
        signature_backref = retval_backref._templatey_signature
        signature_fordref = retval_fordref._templatey_signature

        assert len(signature_backref.dynamic_class_slot_names) == 0
        assert len(signature_fordref.dynamic_class_slot_names) == 0

        # Hard-coding the expected tree is a LOT of tedious manual work, so
        # instead we're just going to be as pragmatic as possible
        dycls_root_backref = signature_backref._dynamic_class_slot_tree
        dycls_root_fordref = signature_fordref._dynamic_class_slot_tree
        assert dycls_root_backref.is_equivalent(dycls_root_fordref)
        assert {slot.slot_path for slot in dycls_root_backref} == {
            ('slot1', Foo),
            ('slot1', Bar),
            ('slot2', Foo),
            ('slot2', Bar),}

        assert (
            dycls_root_backref.get_route_for('slot1', Foo).subtree
                .get_route_for('bar_or_baz', Bar).subtree
        ).dynamic_class_slot_names == {'dynamico'}
        # Baz has no dynamic class slots; it must not be included
        assert not (
            dycls_root_backref.get_route_for('slot1', Foo).subtree
        ).has_route_for('bar_or_baz', Baz)
        assert (
            dycls_root_backref.get_route_for('slot1', Foo).subtree
                .get_route_for('bar_or_baz', Bar).subtree
                    .get_route_for('foo', Foo).subtree
                        .get_route_for('bar_or_baz', Bar).subtree
        ).dynamic_class_slot_names == {'dynamico'}
