from dataclasses import FrozenInstanceError
from dataclasses import is_dataclass
from typing import cast
from typing import get_type_hints

import pytest

from templatey.templates import Content
from templatey.templates import Slot
from templatey.templates import TemplateIntersectable
from templatey.templates import Var
from templatey.templates import is_template_class
from templatey.templates import is_template_instance
from templatey.templates import make_template_definition
from templatey.templates import template

from templatey_testutils import fake_template_config


class TestIsTemplateClass:

    def test_positive(self):
        class FakeTemplate:
            ...

        # Quick and dirty. This will break if we add anything that isn't a
        # class var to the TemplateIntersectable class.
        for key in get_type_hints(TemplateIntersectable):
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
        for key in get_type_hints(TemplateIntersectable):
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

    def test_config_and_locator_assigned(self):
        """The passed template config must be stored on the class, along
        with the template locator.
        """

        class FakeTemplate:
            bar: Var[str]

        retval = cast(TemplateIntersectable, make_template_definition(
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

        retval = cast(TemplateIntersectable, make_template_definition(
            FakeTemplate,
            dataclass_kwargs={},
            template_resource_locator=object(),
            template_config=fake_template_config))
        signature = retval._templatey_signature

        assert len(signature.slots) == 1
        assert 'foo' in signature.slots

    def test_var_extraction(self):
        """Fields declared with Var[...] must be correctly detected
        and stored on the class.
        """
        class FakeTemplate:
            bar: Var[str]
            baz: Content[str]

        retval = cast(TemplateIntersectable, make_template_definition(
            FakeTemplate,
            dataclass_kwargs={},
            template_resource_locator=object(),
            template_config=fake_template_config))
        signature = retval._templatey_signature

        assert len(signature.vars_) == 1
        assert 'bar' in signature.vars_

    def test_content_extraction(self):
        """Fields declared with Content[...] must be correctly detected
        and stored on the class.
        """
        class FakeTemplate:
            bar: Var[str]
            baz: Content[str]

        retval = cast(TemplateIntersectable, make_template_definition(
            FakeTemplate,
            dataclass_kwargs={},
            template_resource_locator=object(),
            template_config=fake_template_config))
        signature = retval._templatey_signature

        assert len(signature.content) == 1
        assert 'baz' in signature.content

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
