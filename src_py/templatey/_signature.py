from __future__ import annotations

import itertools
from collections import defaultdict
from collections import deque
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from types import UnionType
from weakref import ref

from templatey._forwardrefs import PENDING_FORWARD_REFS
from templatey._forwardrefs import ForwardReferenceProxyClass
from templatey._forwardrefs import ForwardRefLookupKey
from templatey._forwardrefs import is_forward_reference_proxy
from templatey._provenance import TemplateProvenance
from templatey._provenance import TemplateProvenanceNode
from templatey._slot_tree import ConcreteSlotTreeNode
from templatey._slot_tree import PendingSlotTreeContainer
from templatey._slot_tree import PendingSlotTreeNode
from templatey._slot_tree import SlotTreeNode
from templatey._slot_tree import SlotTreeRoute
from templatey._slot_tree import merge_into_slot_tree
from templatey._slot_tree import update_encloser_with_trees_from_slot
from templatey._types import BACKSTOP_TEMPLATE_INSTANCE_ID
from templatey._types import TemplateClass
from templatey._types import TemplateInstanceID
from templatey._types import TemplateParamsInstance
from templatey.parser import InterpolatedFunctionCall
from templatey.parser import ParsedTemplateResource

type GroupedTemplateInvocations = dict[TemplateClass, list[TemplateProvenance]]
type TemplateLookupByID = dict[TemplateInstanceID, TemplateParamsInstance]
# Note that there's no need for an abstract version of this, at least not right
# now, because in order to calculate it, we'd need to know the template body,
# which doesn't happen until we already know the template instance, which means
# we can skip ahead to the concrete version.
type EnvFuncInvocation = tuple[TemplateProvenance, InterpolatedFunctionCall]
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

    slot_names: frozenset[str]
    var_names: frozenset[str]
    content_names: frozenset[str]

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

        # Oh thank god.
        tree_wip.default_factory = None
        return cls(
            _forward_ref_lookup_key=forward_ref_lookup_key,
            template_cls_ref=ref(template_cls),
            slot_names=slot_names,
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

    def extract_function_invocations(
            self,
            root_template_instance: TemplateParamsInstance,
            template_preload: dict[TemplateClass, ParsedTemplateResource]
            ) -> list[EnvFuncInvocation]:
        """Looks at all included abstract function invocations, and
        generates lists of their concrete invocations, based on both the
        actual values of slots at the template instances, as well as the
        template definition provided in template_preload.
        """
        invocations: list[EnvFuncInvocation] = []
        root_template_cls = type(root_template_instance)
        all_slot_tree_items: Iterable[tuple[TemplateClass, SlotTreeNode]]
        slot_tree_lookup = self._slot_tree_lookup
        if root_template_cls in slot_tree_lookup:
            all_slot_tree_items = slot_tree_lookup.items()

        else:
            all_slot_tree_items = itertools.chain(
                slot_tree_lookup.items(),
                [(root_template_cls, SlotTreeNode(is_terminus=True))])

        # Keep in mind that the slot tree contains all included slot classes
        # (recursively), not just the ones at the root_template_instance.
        # Our goal here is:
        # 1.. find all template classes with abstract function calls
        # 2.. build provenances for all invocations of those template classes
        # 3.. combine (product) those provenances with all of the abstract
        #     function calls at that template class
        print('\n$$$$$ About to extract')
        for template_class, slot_tree_root in all_slot_tree_items:
            print(f'    checking {template_class}...')
            abstract_calls = template_preload[template_class].function_calls

            # Constructing a provenance is relatively expensive, so we only
            # want to do it if we actually have some function calls within the
            # template
            if abstract_calls:
                print('    abstract calls found')
                provenances = TemplateProvenance.from_slot_tree(
                    root_template_instance,
                    slot_tree_root)
                print(f'    {provenances=}')

                # Oh the humanity, oh the combinatorics!
                invocations.extend(itertools.product(
                    provenances,
                    itertools.chain.from_iterable(abstract_calls.values())))

        print('\n!!!!!!!!!!!!!!!')
        print(invocations)
        print('!!!!!!!!!!!!!!!\n')
        return invocations

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
