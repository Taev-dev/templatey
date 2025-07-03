from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field
from types import UnionType
from weakref import ref

from templatey._forwardrefs import PENDING_FORWARD_REFS
from templatey._forwardrefs import ForwardReferenceProxyClass
from templatey._forwardrefs import ForwardRefLookupKey
from templatey._forwardrefs import is_forward_reference_proxy
from templatey._provenance import Provenance
from templatey._slot_tree import ConcreteSlotTreeNode
from templatey._slot_tree import DynamicClassSlotTreeNode
from templatey._slot_tree import PendingSlotTreeContainer
from templatey._slot_tree import PendingSlotTreeNode
from templatey._slot_tree import SlotTreeNode
from templatey._slot_tree import SlotTreeRoute
from templatey._slot_tree import gather_dynamic_class_slots
from templatey._slot_tree import merge_into_slot_tree
from templatey._slot_tree import update_encloser_with_trees_from_slot
from templatey._types import TemplateClass
from templatey._types import TemplateInstanceID
from templatey._types import TemplateParamsInstance

type GroupedTemplateInvocations = dict[TemplateClass, list[Provenance]]
type TemplateLookupByID = dict[TemplateInstanceID, TemplateParamsInstance]
type _SlotAnnotation = (
    TemplateClass
    | UnionType
    | type[ForwardReferenceProxyClass])


@dataclass(slots=True)
class TemplateSignature:
    """This class stores the processed interface based on the params.
    It gets used to compare with the TemplateParse to make sure that
    the two can be used together.

    Not meant to be created directly; instead, you should use the
    TemplateSignature.new() convenience method.
    """
    # It's nice to have this available, especially when resolving forward refs,
    # but unlike eg the slot tree, it's trivially easy for us to avoid GC
    # loops within the signature
    template_cls_ref: ref[TemplateClass]
    _forward_ref_lookup_key: ForwardRefLookupKey

    # Note that these are all ONLY the direct nesteds; these do not include
    # anything from deeper in the slot tree.
    slot_names: frozenset[str]
    var_names: frozenset[str]
    content_names: frozenset[str]
    dynamic_class_slot_names: frozenset[str]

    _dynamic_class_slot_tree: DynamicClassSlotTreeNode
    # Note that these contain all included types, not just the ones on the
    # outermost layer that are associated with the signature. In other words,
    # they include the flattened recursion of all included slots, all the way
    # down the tree
    _slot_tree_lookup: dict[TemplateClass, ConcreteSlotTreeNode]
    _pending_ref_lookup: dict[ForwardRefLookupKey, PendingSlotTreeContainer]

    # I really don't like that we need to remember to recalculate this every
    # time we update the slot tree lookup, but for rendering performance
    # reasons we want this to be precalculated before every call to render.
    included_template_classes: frozenset[TemplateClass] = field(init=False)

    def __post_init__(self):
        self.refresh_included_template_classes_snapshot()
        self.refresh_pending_forward_ref_registration()

    def refresh_included_template_classes_snapshot(self):
        """Call this when resolving forward references to apply any
        changes made to the slot tree to the template classes snapshot
        we use for increased render performance.
        """
        template_cls = self.template_cls_ref()
        if template_cls is None:
            raise RuntimeError(
                'Template class was garbage collected before template '
                + 'signature, and then signature asked to refresh included '
                + 'classes snapshot?!')

        self.included_template_classes = frozenset(
            {template_cls, *self._slot_tree_lookup})

    def refresh_pending_forward_ref_registration(self):
        """Call this after having resolved forward references (or when
        initially constructing the template signature) to register the
        template class as requiring its forward refs. This is what
        plumbs up the notification code to actually initiate resolving.
        """
        template_cls = self.template_cls_ref()
        if template_cls is None:
            raise RuntimeError(
                'Template class was garbage collected before template '
                + 'signature, and then signature asked to refresh pending '
                + 'forward ref registration?!')

        forward_ref_registry = PENDING_FORWARD_REFS.get()
        for forward_ref in self._pending_ref_lookup:
            forward_ref_registry[forward_ref].add(template_cls)

    @classmethod
    def new(
            cls,
            template_cls: type,
            slots: dict[str, _SlotAnnotation],
            dynamic_class_slot_names: set[str],
            vars_: dict[str, type | type[ForwardReferenceProxyClass]],
            content: dict[str, type | type[ForwardReferenceProxyClass]],
            *,
            forward_ref_lookup_key: ForwardRefLookupKey
            ) -> TemplateSignature:
        """Create a new TemplateSignature based on the gathered slots,
        vars, and content. This does all of the convenience calculations
        needed to populate the semi-redundant fields.
        """
        slot_names = frozenset(slots)
        var_names = frozenset(vars_)
        content_names = frozenset(content)

        # Quick refresher: our goal here is to construct a lookup that gets
        # us a route to every instance of a particular template type. In other
        # words, we want to be able to check a template type, and then see all
        # possible getattr() sequences that arrive at an instance of that
        # template type.
        tree_wip: dict[TemplateClass, ConcreteSlotTreeNode]
        tree_wip = defaultdict(SlotTreeNode)
        pending_ref_lookup: \
            dict[ForwardRefLookupKey, PendingSlotTreeContainer] = {}

        concrete_slot_defs, pending_ref_defs = cls._normalize_slot_defs(
            slots.items())

        # Note that order between concrete_slot_defs and pending_ref_defs
        # DOES matter here. Concrete has to come first, because we need to
        # discover all reference loops back to the template class before
        # we can fully define the pending trees.
        for slot_name, slot_type in concrete_slot_defs:
            # In the simple recursion case -- a template defines a slot of its
            # own class -- we can immediately create a reference loop without
            # any pomp nor circumstance.
            # Also: yes, this is resolved at annotation time, and not a forward
            # ref!
            if slot_type is template_cls:
                recursive_slot_tree = tree_wip[slot_type]
                recursive_slot_tree.is_terminus = True
                recursive_slot_tree.is_recursive = True
                recursive_slot_route = SlotTreeRoute.new(
                    slot_name,
                    slot_type,
                    recursive_slot_tree)
                recursive_slot_tree.append(recursive_slot_route)

            # Remember that we expanded the union already, so this is
            # guaranteed to be a single concrete ``slot_type``.
            else:
                offset_tree = PendingSlotTreeNode(
                    insertion_slot_names={slot_name})
                update_encloser_with_trees_from_slot(
                    template_cls,
                    tree_wip,
                    pending_ref_lookup,
                    forward_ref_lookup_key,
                    slot_type,
                    offset_tree,)

        # Note that by "direct" we mean, immediate nested children of the
        # current template_cls, for which we're constructing a signature,
        # and NOT any nested children of nested concrete slots.
        # These plain pending refs are very straightforward, since we don't
        # know anything about them yet (by definition; they're forward refs!).
        for (
            direct_pending_slot_name,
            direct_forward_ref_lookup_key
        ) in pending_ref_defs:
            existing_pending_tree = pending_ref_lookup.get(
                direct_forward_ref_lookup_key)

            if existing_pending_tree is None:
                dest_insertion = PendingSlotTreeNode(
                    insertion_slot_names={direct_pending_slot_name})
                pending_ref_lookup[direct_forward_ref_lookup_key] = (
                    PendingSlotTreeContainer(
                        pending_slot_type=direct_forward_ref_lookup_key,
                        pending_root_node=dest_insertion))
                # Note that we need to include any recursion loops that end
                # up back at the template class, since they would ALSO have
                # the same insertion points. Helpfully, we can just merge in
                # any existing tree for that.
                existing_recursive_self_tree = tree_wip.get(
                    template_cls,
                    SlotTreeNode())
                merge_into_slot_tree(
                    template_cls,
                    None,
                    dest_insertion,
                    existing_recursive_self_tree)

            else:
                (existing_pending_tree
                    .pending_root_node
                    .insertion_slot_names.add(direct_pending_slot_name))

        # Okay, so here's the deal. Every time we add a slot,
        # might create a recursive reference loop (if it had a forward ref to
        # the current enclosing class). So far so good. The problem
        # is, this means that existing slot trees will be incomplete.
        # Therefore, at the end of the process, we always circle back and check
        # for a self-referential slot tree, and if it exists, merge it into
        # all the rest.
        self_referential_recursive_tree = tree_wip.get(template_cls)
        if self_referential_recursive_tree is not None:
            for slot_type, slot_tree in tree_wip.items():
                if slot_type is not template_cls:
                    merge_into_slot_tree(
                        template_cls,
                        slot_type,
                        slot_tree,
                        self_referential_recursive_tree)

        # Oh thank god.
        tree_wip.default_factory = None
        return cls(
            _forward_ref_lookup_key=forward_ref_lookup_key,
            template_cls_ref=ref(template_cls),
            slot_names=slot_names,
            dynamic_class_slot_names=frozenset(dynamic_class_slot_names),
            _dynamic_class_slot_tree=gather_dynamic_class_slots(
                template_cls,
                dynamic_class_slot_names,
                tree_wip),
            var_names=var_names,
            content_names=content_names,
            _slot_tree_lookup=tree_wip,
            _pending_ref_lookup=pending_ref_lookup)

    @classmethod
    def _normalize_slot_defs(
            cls,
            slot_defs: Iterable[tuple[str, _SlotAnnotation]]
            ) -> tuple[
                list[tuple[str, TemplateClass]],
                list[tuple[str, ForwardRefLookupKey]]]:
        """The annotations we get "straight off the tap" (so to speak)
        of the template class can be:
        ++  unions
        ++  type aliases, though we don't quite support these yet
            (though this function should make it relatively
            straightforward to do so)
        ++  concrete backrefs to existing templates
        ++  pending forward refs to not-yet-defined templates

        This function is responsible for giving us a clean, uniform way
        of representing them. First, it expands all unions etc into
        multiple slot paths. Second, it splits the concrete from the
        pending nodes, returning them separately.
        """
        concrete_slots = []
        pending_refs = []

        flattened_defs: \
            list[tuple[
                str,
                TemplateClass | type[ForwardReferenceProxyClass]]] = []
        for slot_name, slot_annotation in slot_defs:
            # Note that this still might contain a heterogeneous mix of
            # template classes and forward refs! Hence flattening first.
            if isinstance(slot_annotation, UnionType):
                for slot_type in slot_annotation.__args__:
                    flattened_defs.append((slot_name, slot_type))

            else:
                flattened_defs.append((slot_name, slot_annotation))

        # Again, this looks redundant at first glance, but the point was to
        # normalize unions into single types, whether concrete or pending
        for slot_name, flattened_slot_annotation in flattened_defs:
            if is_forward_reference_proxy(flattened_slot_annotation):
                forward_ref = flattened_slot_annotation.REFERENCE_TARGET
                pending_refs.append((slot_name, forward_ref))

            else:
                concrete_slots.append((slot_name, flattened_slot_annotation))

        return concrete_slots, pending_refs

    def resolve_forward_ref(
            self,
            lookup_key: ForwardRefLookupKey,
            resolved_template_cls: TemplateClass
            ) -> None:
        """Notifies a dependent class (one that declared a slot as a
        forward reference) that the reference is now available, thereby
        causing it to resolve the forward ref and remove it from its
        pending trees.
        """
        enclosing_template_cls = self.template_cls_ref()
        if enclosing_template_cls is None:
            raise RuntimeError(
                'Template class was garbage collected before template '
                + 'signature, and then signature asked to resolve forward '
                + 'ref?!')

        update_encloser_with_trees_from_slot(
            enclosing_template_cls,
            self._slot_tree_lookup,
            self._pending_ref_lookup,
            self._forward_ref_lookup_key,
            resolved_template_cls,
            self._pending_ref_lookup.pop(lookup_key).pending_root_node,)

        self.refresh_included_template_classes_snapshot()
        self.refresh_pending_forward_ref_registration()

        # Okay, so here's the deal. Every time we resolve a forward ref, we
        # might create a recursive reference loop. So far so good. The problem
        # is, this means that existing slot trees will be incomplete.
        # therefore, at the end of the process, we always circle back and check
        # for a self-referential slot tree, and if it exists, merge it into
        # all the rest.
        self_referential_recursive_tree = self._slot_tree_lookup.get(
            enclosing_template_cls)
        if self_referential_recursive_tree is not None:
            for slot_type, slot_tree in self._slot_tree_lookup.items():
                if slot_type is not enclosing_template_cls:
                    merge_into_slot_tree(
                        enclosing_template_cls,
                        slot_type,
                        slot_tree,
                        self_referential_recursive_tree)

        # Right, so the thing is, this might have introduced a bunch of new
        # recursive reference loops, and those might be buried within several
        # layers of pending trees and stuff. This isn't suuuuper performance
        # critical, so by far the easiest thing to do is just rebuild the
        # tree in its entirety.
        # Note: I suppose in theory we could maybe just do the same as we're
        # doing for the normal slot trees, and just constantly re-merge any
        # self_referential_recursive_tree?
        self._dynamic_class_slot_tree = gather_dynamic_class_slots(
            enclosing_template_cls,
            set(self.dynamic_class_slot_names),
            self._slot_tree_lookup)

    def stringify_all(self) -> str:
        """This is a debug method that creates a prettified string
        version of the entire slot tree lookup (pending and concrete).
        """
        to_join = []
        to_join.append('Resolved (concrete) slots:')
        for template_class, root_node in self._slot_tree_lookup.items():
            to_join.append(f'  {template_class}')
            to_join.append(root_node.stringify(depth=1))

        to_join.append('Pending (forward reference) slots:')
        for ref_lookup_key, container in self._pending_ref_lookup.items():
            to_join.append(f'  {ref_lookup_key}')
            to_join.append(container.pending_root_node.stringify(depth=1))

        return '\n'.join(to_join)
