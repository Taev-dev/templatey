from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

from templatey._types import TemplateClass
from templatey._types import TemplateInstanceID
from templatey._types import TemplateParamsInstance
from templatey.parser import NestedContentReference
from templatey.parser import NestedVariableReference
from templatey.parser import ParsedTemplateResource


@dataclass(slots=True, frozen=True)
class TemplateProvenanceNode:
    """TemplateProvenanceNode instances are unique to the exact location
    on the exact render tree of a particular template instance. If the
    template instance gets reused within the same render tree, it will
    have multiple provenance nodes. And each different slot in an
    enclosing template will have a separate provenance node, potentially
    with different namespace overrides.

    This is used both during function execution (to calculate the
    concrete value of any parameters), and also while flattening the
    render tree.

    Also note that the root template, **along with any templates
    injected into the render by environment functions,** will have an
    empty list of provenance nodes.

    Note that, since overrides from the enclosing template can come
    exclusively from the template body -- and are therefore shared
    across all nested children of the same slot -- they don't get stored
    within the provenance, since we'd require access to the template
    bodies, which we don't yet have.
    """
    encloser_slot_key: str
    encloser_slot_index: int
    # The reason to have both the instance and the instance ID is so that we
    # can have hashability of the ID while not imposing an API on the instances
    instance_id: TemplateInstanceID
    instance: TemplateParamsInstance = field(compare=False)


class TemplateProvenance(tuple[TemplateProvenanceNode]):

    def bind_content(
            self,
            name: str,
            template_preload: dict[TemplateClass, ParsedTemplateResource]
            ) -> object:
        """Use this to calculate a concrete value for use in rendering.
        This walks up the provenance stack, recursively looking for any
        overrides to the content. If none are found, it returns the
        value from the childmost instance in the provenance.
        """
        # We use the literal ellipsis type as a sentinel for values not being
        # added yet, so we might as well just continue the trend!
        current_provenance_node = self[-1]
        value = getattr(current_provenance_node.instance, name, ...)
        encloser_param_name = name
        encloser_slot_key = current_provenance_node.encloser_slot_key
        for encloser in reversed(self[0:-1]):
            template_class = type(encloser.instance)
            # We do this so that env funcs that inject templates don't try
            # to continue looking up the provenance tree for slots that don't
            # actually exist.
            if hasattr(template_class, '_TEMPLATEY_VIRTUAL_INSTANCE'):
                break

            encloser_template = template_preload[template_class]
            encloser_overrides = (
                encloser_template.slots[encloser_slot_key].params)

            if encloser_param_name in encloser_overrides:
                value = encloser_overrides[encloser_param_name]

                if isinstance(value, NestedContentReference):
                    encloser_slot_key = encloser.encloser_slot_key
                    encloser_param_name = value.name
                    value = ...
                else:
                    break
            else:
                break

        if value is ...:
            raise KeyError(
                'No value found for content with name at slot!',
                self[-1].instance, name)

        return value

    def bind_variable(
            self,
            name: str,
            template_preload: dict[TemplateClass, ParsedTemplateResource]
            ) -> object:
        """Use this to calculate a concrete value for use in rendering.
        This walks up the provenance stack, recursively looking for any
        overrides to the variable. If none are found, it returns the
        value from the childmost instance in the provenance.
        """
        # We use the literal ellipsis type as a sentinel for values not being
        # added yet, so we might as well just continue the trend!
        current_provenance_node = self[-1]
        value = getattr(current_provenance_node.instance, name, ...)
        encloser_param_name = name
        encloser_slot_key = current_provenance_node.encloser_slot_key
        for encloser in reversed(self[0:-1]):
            template_class = type(encloser.instance)
            # We do this so that env funcs that inject templates don't try
            # to continue looking up the provenance tree for slots that don't
            # actually exist.
            if hasattr(template_class, '_TEMPLATEY_VIRTUAL_INSTANCE'):
                break

            encloser_template = template_preload[template_class]
            encloser_overrides = (
                encloser_template.slots[encloser_slot_key].params)

            if encloser_param_name in encloser_overrides:
                value = encloser_overrides[encloser_param_name]

                if isinstance(value, NestedVariableReference):
                    encloser_slot_key = encloser.encloser_slot_key
                    encloser_param_name = value.name
                    value = ...
                else:
                    break
            else:
                break

        if value is ...:
            raise KeyError(
                'No value found for variable with name at slot!',
                self[-1].instance, name)

        return value
