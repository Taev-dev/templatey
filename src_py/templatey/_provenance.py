from __future__ import annotations

from collections.abc import Sequence
from dataclasses import KW_ONLY
from dataclasses import dataclass
from dataclasses import field

from templatey._slot_tree import SlotTreeNode
from templatey._types import TemplateClass
from templatey._types import TemplateInstanceID
from templatey._types import TemplateParamsInstance
from templatey.parser import NestedContentReference
from templatey.parser import NestedVariableReference
from templatey.parser import ParsedTemplateResource


@dataclass(slots=True, frozen=True)
class ProvenanceNode:
    """ProvenanceNode instances are unique to the exact location
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
    instance: TemplateParamsInstance = field(compare=False, repr=False)


@dataclass(slots=True, frozen=True)
class Provenance:
    slotpath: tuple[ProvenanceNode, ...] = field(default=())
    _: KW_ONLY
    from_injection: Provenance | None = None

    # Used for memoization
    _hash: int | None = field(
        default=None, compare=False, init=False, repr=False)

    def with_appended(self, node_to_append: ProvenanceNode) -> Provenance:
        return Provenance(
            (*self.slotpath, node_to_append),
            from_injection=self.from_injection)

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
        slotpath = self.slotpath
        # We use the literal ellipsis type as a sentinel for values not being
        # added yet, so we might as well just continue the trend!
        current_provenance_node = slotpath[-1]
        value = getattr(current_provenance_node.instance, name, ...)
        encloser_param_name = name
        encloser_slot_key = current_provenance_node.encloser_slot_key
        for encloser in reversed(slotpath[0:-1]):
            template_class = type(encloser.instance)
            # We do this so that env funcs that inject templates don't try
            # to continue looking up the provenance tree for slots that don't
            # actually exist.
            if hasattr(template_class, '_TEMPLATEY_EMPTY_INSTANCE'):
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
                slotpath[-1].instance, name)

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
        slotpath = self.slotpath
        # We use the literal ellipsis type as a sentinel for values not being
        # added yet, so we might as well just continue the trend!
        current_provenance_node = slotpath[-1]
        value = getattr(current_provenance_node.instance, name, ...)
        encloser_param_name = name
        encloser_slot_key = current_provenance_node.encloser_slot_key
        for encloser in reversed(slotpath[0:-1]):
            template_class = type(encloser.instance)
            # We do this so that env funcs that inject templates don't try
            # to continue looking up the provenance tree for slots that don't
            # actually exist.
            if hasattr(template_class, '_TEMPLATEY_EMPTY_INSTANCE'):
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
                slotpath[-1].instance, name)

        return value

    @classmethod
    def from_slot_tree(
            cls,
            root_template_instance: TemplateParamsInstance,
            root_slot_tree: SlotTreeNode,
            from_injection: Provenance | None,
            ) -> list[Provenance]:
        """Given a root template instance, walks the passed slot tree,
        finding all terminus points. Returns them as a list of
        provenance instances.

        Note that you must choose the correct slot tree from the
        slot tree lookup in advance; this simply converts slot tree
        terminus points into provenances based on a root template
        instance: no more, no less.

        TODO: it would be nice if we could pre-merge the slot trees during
        template loading for the actual use cases (env functions invocations
        and, eventually, dynamic slot types) so that we didn't need a separate
        tree traversal for every template class. But for now, this is good
        enough
        """
        provenances: list[Provenance] = []
        stack: list[_TreeFlattenerFrame] = [
            _TreeFlattenerFrame(
                active_instance=root_template_instance,
                active_subtree=root_slot_tree,
                target_subtree_index=0,
                target_instance_index=0,
                wip_provenance=Provenance(
                    (
                        ProvenanceNode(
                            encloser_slot_key='',
                            encloser_slot_index=-1,
                            instance_id=id(root_template_instance),
                            instance=root_template_instance),),
                    from_injection=from_injection))]

        while stack:
            this_frame = stack[-1]
            if this_frame.exhausted:
                stack.pop()

                if this_frame.active_subtree.is_terminus:
                    provenances.append(this_frame.wip_provenance)

                continue

            this_slot_route = this_frame.active_subtree[
                this_frame.target_subtree_index]
            this_slot_name, this_slot_type, this_subtree = this_slot_route
            target_instance_index = this_frame.target_instance_index

            # This is, in a way, a nested stack, but we're maintaining
            # the stack state within the _TreeFlattenerFrame.
            # At any rate, we use the zero-index iteration of the loop
            # to memoize some values on the stack frame.
            if target_instance_index == 0:
                target_instances = getattr(
                    this_frame.active_instance,
                    this_slot_name)
                target_instances_count = len(target_instances)

                # Check in advance if there are no target instances at all,
                # and if so, skip the whole thing. This isn't just for
                # performance; the processing logic depends on it.
                if target_instances_count > 0:
                    this_frame.target_instances_count = target_instances_count
                    this_frame.target_instances = target_instances
                else:
                    # Note: this is critical! Otherwise we'll infinitely loop.
                    this_frame.target_subtree_index += 1
                    continue

            else:
                target_instances_count = this_frame.target_instances_count
                # We've exhausted the target instances; reset the state for
                # the next slot tree route and then continue.
                if this_frame.target_instance_index >= target_instances_count:
                    # Note: we're deliberately skipping the target instances
                    # themselves, because it'll just get overwritten the next
                    # time around, so we can save ourselves an operation.
                    this_frame.target_instances_count = 0
                    this_frame.target_instance_index = 0
                    # Note: this is critical! Otherwise we'll infinitely loop.
                    this_frame.target_subtree_index += 1
                    continue

                # We still have some instances to target; normalize the state
                # so that we can operate on them.
                target_instances = this_frame.target_instances

            # Okay, status check: we have our stack frame state configured
            # correctly, and we have target instances to check.
            instance_to_check = target_instances[target_instance_index]
            # Note: exact match here; not subclassing! Subclassing breaks too
            # many things, so we don't support it.
            if type(instance_to_check) is this_slot_type:
                stack.append(_TreeFlattenerFrame(
                    active_instance=instance_to_check,
                    active_subtree=this_subtree,
                    target_subtree_index=0,
                    target_instance_index=0,
                    wip_provenance=Provenance(
                        (
                            *this_frame.wip_provenance.slotpath,
                            ProvenanceNode(
                                encloser_slot_key=this_slot_name,
                                encloser_slot_index=target_instance_index,
                                instance_id=id(instance_to_check),
                                instance=instance_to_check)),
                        from_injection=from_injection)))

            this_frame.target_instance_index += 1

        return provenances

    def __hash__(self) -> int:
        memoized = self._hash
        if memoized is None:
            retval = hash(self.slotpath) ^ hash(self.from_injection)
            object.__setattr__(self, '_hash', retval)
            return retval

        else:
            return memoized


@dataclass(slots=True)
class _TreeFlattenerFrame:
    active_instance: TemplateParamsInstance
    active_subtree: SlotTreeNode
    target_subtree_index: int
    target_instance_index: int
    target_instances_count: int = field(kw_only=True, default=0)
    target_instances: Sequence[TemplateParamsInstance] = field(
        kw_only=True, init=False)
    wip_provenance: Provenance

    _active_subtree_len: int = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        self._active_subtree_len = len(self.active_subtree)

    @property
    def exhausted(self) -> bool:
        return self.target_subtree_index >= self._active_subtree_len
