from typing import cast

from templatey._slot_tree import DynamicClassSlotTreeNode
from templatey._slot_tree import extract_dynamic_class_slot_types
from templatey._types import DynamicClassSlot
from templatey._types import TemplateIntersectable
from templatey._types import TemplateParamsInstance
from templatey._types import Var
from templatey.templates import template

from templatey_testutils import fake_template_config


class TestExtractDynamicClassSlotTypes:

    def test_recursive_dynamic_classes(self):
        """Dynamic-class slots, which have instances that themselves
        declare dynamic-class slots, must result in all slot types
        being successfully loaded.
        """
        @template(fake_template_config, object())
        class Outermost:
            middle: DynamicClassSlot

        @template(fake_template_config, object())
        class Middle:
            innermost: DynamicClassSlot

        @template(fake_template_config, object())
        class Innermost:
            var: Var[str]

        # We want to isolate this test from the construction of the dynamic
        # class slot tree, so we're constructing an explicit one here, and
        # manually updating the classes with it. This will break if we update
        # the struture of the dynamic class slot tree, but it gives us much
        # better test specificity, which is the whole reason we're writing
        # this particular unit test.
        outermost_dynacls_slot_tree = DynamicClassSlotTreeNode(
            dynamic_class_slot_names={'middle'})
        middle_dynacls_slot_tree = DynamicClassSlotTreeNode(
            dynamic_class_slot_names={'innermost'})
        outermost_xable = cast(type[TemplateIntersectable], Outermost)
        middle_xable = cast(type[TemplateIntersectable], Middle)
        outermost_xable._templatey_signature._dynamic_class_slot_tree = (
            outermost_dynacls_slot_tree)
        middle_xable._templatey_signature._dynamic_class_slot_tree = (
            middle_dynacls_slot_tree)

        instance_tree = Outermost(
            middle=[
                Middle(
                    innermost=[Innermost(var='foo')])])

        result = extract_dynamic_class_slot_types(
            instance_tree, outermost_dynacls_slot_tree)

        assert result == {Middle, Innermost}

    def test_similar_recursion_new_instance(self):
        """As long as it encounters new instances (and not a recursive
        self; see below), recursion must continue to check for more
        dynamic-class instances, period, regardless of whether or not
        the current class was already found.
        """
        @template(fake_template_config, object())
        class RecursiveTwinner:
            inner: DynamicClassSlot

        @template(fake_template_config, object())
        class Inner1:
            var: Var[str]

        @template(fake_template_config, object())
        class Inner2:
            var: Var[str]

        # We want to isolate this test from the construction of the dynamic
        # class slot tree, so we're constructing an explicit one here, and
        # manually updating the classes with it. This will break if we update
        # the struture of the dynamic class slot tree, but it gives us much
        # better test specificity, which is the whole reason we're writing
        # this particular unit test.
        twinner_dynacls_slot_tree = DynamicClassSlotTreeNode(
            dynamic_class_slot_names={'inner'})
        twinner_xable = cast(type[TemplateIntersectable], RecursiveTwinner)
        twinner_xable._templatey_signature._dynamic_class_slot_tree = (
            twinner_dynacls_slot_tree)

        instance_tree = RecursiveTwinner(
            inner=[
                RecursiveTwinner(
                    inner=[Inner1(var='foo'), Inner2(var='bar')])])

        result = extract_dynamic_class_slot_types(
            instance_tree, twinner_dynacls_slot_tree)

        # Note that RecursiveTwinner gets included here, but then filtered out
        # by the prepopulation method. This is expected; it lets us skip a
        # repeated check if the current instance is already known and batch
        # them at the end using set operations.
        assert result == {RecursiveTwinner, Inner1, Inner2}

    def test_recursive_self(self):
        """A dynamic-class slot that contains itself as a member must
        not cause an infinite loop of dynamic class extraction (even
        though it would fail to render).

        This is primarily meant to ensure that we're not doing extra
        work when extracting dynamic slots when duplicate, non-recursive
        dynamic instances are passed, but it's an easy way to test for
        that case.
        """
        @template(fake_template_config, object())
        class Recursor:
            recursor: DynamicClassSlot

        @template(fake_template_config, object())
        class Bystander:
            var: Var[str]

        # We want to isolate this test from the construction of the dynamic
        # class slot tree, so we're constructing an explicit one here, and
        # manually updating the classes with it. This will break if we update
        # the struture of the dynamic class slot tree, but it gives us much
        # better test specificity, which is the whole reason we're writing
        # this particular unit test.
        recursor_dynacls_slot_tree = DynamicClassSlotTreeNode(
            dynamic_class_slot_names={'recursor'})
        recursor_xable = cast(type[TemplateIntersectable], Recursor)
        recursor_xable._templatey_signature._dynamic_class_slot_tree = (
            recursor_dynacls_slot_tree)

        instance_list: list[TemplateParamsInstance] = [Bystander(var='foo')]
        instance = Recursor(recursor=instance_list)
        # This tests direct recursion to self
        instance_list.append(instance)
        # This tests indirect recursion to self
        instance_list.append(Recursor(recursor=[instance]))

        result = extract_dynamic_class_slot_types(
            instance, recursor_dynacls_slot_tree)

        # Note that the recursor here is coming strictly from INDIRECT
        # recursion; if we only had direct recursion, it would be empty.
        assert result == {Bystander, Recursor}
