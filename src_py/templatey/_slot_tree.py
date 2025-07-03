from __future__ import annotations

import operator
from collections.abc import Iterable
from collections.abc import Sequence
from copy import copy
from dataclasses import KW_ONLY
from dataclasses import InitVar
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from typing import Self
from typing import cast
from typing import overload

from templatey._forwardrefs import ForwardRefLookupKey
from templatey._types import TemplateClass
from templatey._types import TemplateIntersectable
from templatey._types import TemplateParamsInstance
from templatey._types import create_templatey_id


# Note: the ordering here is to emphasize the fact that the slot
# name is on the ENCLOSING template, but the slot type is from the
# NESTED template
class SlotTreeRoute[T: SlotTreeNode](tuple[str, TemplateClass, T]):
    """An individual route on the slot tree is defined by the attribute
    name for the slot, the slot type, and the subtree from the slot
    class.

    These are optimized for the non-union case. Traversing the slot tree
    with union types will result in a bunch of unnecessary comparisons
    against slot names of different slot types.

    Note that slot tree routes always have a concrete slot name and slot
    type, regardless of whether they're in a pending or concrete tree.
    The reason is simple: in a pending tree, all of the pending classes
    are dead-end nodes, and define their insertion points using just
    the string of the slot name, and nothing else.
    """
    @classmethod
    def new(
            cls,
            slot_name: str,
            slot_type: TemplateClass,
            subtree: T,
            ) -> SlotTreeRoute[T]:
        return cls((slot_name, slot_type, subtree))

    @property
    def subtree(self) -> T:
        """This is slower than directly accessing the tuple values, but
        it makes for clearer code during tree building, where
        performance isn't quite so critical.
        """
        return self[2]

    @property
    def slot_path(self) -> tuple[str, TemplateClass]:
        return self[0:2]


@dataclass(slots=True)
class SlotTreeNode[T: SlotTreeNode](list[SlotTreeRoute[T]]):
    """The purpose of the slot tree is to precalculate what sequences of
    getattr() calls we need to traverse to arrive at every instance of a
    particular slot type for a given template, including all nested
    templates.

    **These are optimized for rendering, not for template declaration.**
    Also note that these are optimized for slots that are not declared
    as type unions; type unions will result in a number of unnecessary
    comparisons against the routes of the other slot types in the union.

    The reason this is useful is for batching during rendering. This is
    important for function calls: it allows us to pre-execute all env
    func calls for a template before we start rendering it. In the
    future, it will also serve the same role for discovering the actual
    template types for dynamic slots, allowing us to load the needed
    template types in advance.

    An individual node on the slot tree is a list of all possible
    attribute names (as ``SlotTreeRoute``s) that a particular search
    pass needs to check for a given instance. Note that **all** of the
    attributes must be searched -- hence using an iteration-optimized
    list instead of a mapping.
    """
    routes: InitVar[Iterable[SlotTreeRoute[T]] | None] = None

    _: KW_ONLY
    # We use this to limit the number of entries we need in the transmogrifier
    # lookup during tree merging/copying
    is_recursive: bool = False
    # We use this to differentiate between dissimilar unions, one which
    # continues on to a subtree, and one which ends here with the target
    # instance
    is_terminus: bool = False

    id_: int = field(default_factory=create_templatey_id)

    # We use this to make the logic cleaner when merging trees, but we want
    # the faster performance of the tuple when actually traversing the tree
    _index_by_slot_path: dict[tuple[str, TemplateClass], int] = field(
            init=False, repr=False)

    def __post_init__(
            self,
            routes: Iterable[SlotTreeRoute[T]] | None):
        # Explicit instead of super because... idunno, we're breaking things
        # somehow
        if routes is None:
            list.__init__(self)
        else:
            list.__init__(self, routes)

        self._index_by_slot_path = {
            route.slot_path: index for index, route in enumerate(self)}

    def empty_clone(
            self,
            *,
            fields_to_skip: set[str] | frozenset[str] = frozenset()
            ) -> Self:
        """This creates a clone of the node without any routes. Useful
        for merging and copying, where you need to do some manual
        transform of the content.

        Note that this is almost the same as dataclasses.replace, with
        the exception that we create shallow copies of attributes
        instead of preserving them.
        """
        kwargs = {}
        for dc_field in fields(self):
            if dc_field.init and dc_field.name not in fields_to_skip:
                # Note: the copy here is important for any mutable values,
                # notably the insertion_slot_names.
                kwargs[dc_field.name] = copy(getattr(self, dc_field.name))

        return type(self)(**kwargs)

    def merge_fields_only(
            self,
            other: SlotTreeNode,
            *,
            fields_to_skip: set[str] | frozenset[str] = frozenset()):
        """Updates the current node, merging in all non-init field
        values from other, using |=. Only merges values that exist on
        the current node, allowing for transformation between pending
        and concrete node types.

        Leaves the ID of the current node unchanged.
        """
        # Always skip ``id_`` -- always preserve the original ID!
        fields_to_skip = fields_to_skip | {'id_'}
        missing = object()

        for dc_field in fields(self):
            if dc_field.init and dc_field.name not in fields_to_skip:
                current_value = getattr(self, dc_field.name)
                other_value = getattr(other, dc_field.name, missing)

                if other_value is not missing:
                    setattr(
                        self,
                        dc_field.name,
                        operator.ior(current_value, other_value))

    @property
    def requires_transmogrification(self) -> bool:
        """This determines whether or not copies of the tree require
        some post-processing to make sure that the tree STRUCTURE is
        the same. It is used by both copying and merging trees to make
        sure that the transmogrification lookup is as sparse as
        possible.

        For the base class, we simply wrap ``is_recursive``, but for
        the pending tree derived class, we also check other stuff --
        hence the wrapping.
        """
        return self.is_recursive

    def append(self, route: SlotTreeRoute[T]):
        """This has slightly different semantics to normal list
        appending if you're trying to append a duplicate slot path:
        ++  if it's identical to the existing path (ie, it targets the
            same node), we do nothing
        ++  if it's not identical, we error
        """
        slot_path = (route[0], route[1])
        slot_index = self._index_by_slot_path.get(slot_path)

        if slot_index is not None:
            if self[slot_index].subtree is route.subtree:
                return
            else:
                raise ValueError(
                    'Templatey internal error: attempt to append duplicate '
                    + 'slot name for same slot type, but a different subtree! '
                    + 'Please search for / report issue to github along with '
                    + 'a traceback.')

        list.append(self, route)
        self._index_by_slot_path[slot_path] = len(self) - 1

    def has_route_for(
            self,
            slot_name: str,
            slot_type: TemplateClass
            ) -> bool:
        return (slot_name, slot_type) in self._index_by_slot_path

    def get_route_for(
            self,
            slot_name: str,
            slot_type: TemplateClass
            ) -> SlotTreeRoute[T]:
        return self[self._index_by_slot_path[(slot_name, slot_type)]]

    def rewrite_route_for(
            self,
            slot_name: str,
            slot_type: TemplateClass,
            new_route: SlotTreeRoute[T]
            ) -> None:
        dest_index = self._index_by_slot_path[(slot_name, slot_type)]
        list.__setitem__(self, dest_index, new_route)

    def stringify(
            self,
            *,
            depth=0,
            _encountered_ids: frozenset[int] = frozenset()
            ) -> str:
        """Creates a pretty-print-style string representation of the
        node and all its nested nodes, recursively.
        """
        indentation = '    ' * depth

        to_join = []
        for dc_field in fields(self):
            if dc_field.init:
                field_name = dc_field.name
                to_join.append(
                    f'{indentation}{field_name}: {getattr(self, field_name)}')

        if self.id_ in _encountered_ids:
            to_join.append(
                f'{indentation}... Recursion detected; omitting subtrees.')

        else:
            for route in self:
                to_join.append(
                    f'{indentation}++  {route[0:2]}')
                to_join.append(route[2].stringify(
                    depth=depth + 1,
                    _encountered_ids=_encountered_ids | {self.id_}))

        return '\n'.join(to_join)

    def is_equivalent(
            self,
            other: SlotTreeNode,
            *,
            _previous_encounters: dict[int, int] | None = None
            ) -> bool:
        """This compares two slot tree nodes recursively, ignoring IDs.
        If they are otherwise identical -- including the structure of
        any recursive loops -- returns True.

        Note that this is optimized for maintainability and not
        performance. Its primary intended use is in testing.
        """
        previous_encounters: dict[int, int]
        if _previous_encounters is None:
            previous_encounters = {}
        else:
            # Note that the copy here is important because otherwise we'd be
            # mutating state even in other tree branches which might not have
            # encountered us (since we reuse this in the recursive case)
            previous_encounters = {**_previous_encounters}

        # These two cases are trivial: first, if the types aren't the same,
        # they're not equivalent, period. Second, if we already encountered
        # that node's ID, that means we're in a recursive loop, and the two
        # IDs must match.
        if type(self) is not type(other):
            return False
        if other.id_ in previous_encounters:
            return previous_encounters[other.id_] == self.id_

        # First check all the fields, since this should in theory be quick.
        # Yes we could do a bool for this, but doing it as a dict makes it
        # easier to add in temporary print debugs if required
        fields_match: dict[str, bool] = {}
        for dc_field in fields(self):
            field_name = dc_field.name
            if dc_field.init and field_name != 'id_':
                fields_match[field_name] = (
                    getattr(self, field_name) == getattr(other, field_name))
        if not all(fields_match.values()):
            return False

        # Now make sure the routes are the same. This lets us simplify the
        # recursive comparison logic; we don't need any checks to make sure
        # that there weren't any leftover routes on either self or other.
        # It also lets us short-circuit without recursion if they don't match,
        # but that's not the primary motivation behind it.
        self_routes = set(self._index_by_slot_path)
        other_routes = set(other._index_by_slot_path)
        if self_routes != other_routes:
            return False

        # Now finally we're getting to recursive territory.
        previous_encounters[other.id_] = self.id_
        for slot_path in self._index_by_slot_path:
            self_index = self._index_by_slot_path[slot_path]
            self_subtree = self[self_index].subtree
            other_index = other._index_by_slot_path[slot_path]
            other_subtree = other[other_index].subtree

            # Recursion is glorious as long as you don't need to worry about
            # performance! Too bad we can't use it during rendering (because
            # it's way too slow)
            if not self_subtree.is_equivalent(
                other_subtree,
                _previous_encounters=previous_encounters
            ):
                return False

        return True

    def extend(self, routes: Iterable[SlotTreeRoute[T]]):
        raise ZeroDivisionError('Templatey internal error: not implemented!')

    def __delitem__(self, *args, **kwargs):
        raise ZeroDivisionError('Templatey internal error: not implemented!')

    def __setitem__(self, *args, **kwargs):
        raise ZeroDivisionError('Templatey internal error: not implemented!')

    def clear(self, *args, **kwargs):
        raise ZeroDivisionError('Templatey internal error: not implemented!')

    def copy(self, *args, **kwargs):
        raise ZeroDivisionError('Templatey internal error: not implemented!')

    def insert(self, *args, **kwargs):
        raise ZeroDivisionError('Templatey internal error: not implemented!')

    def pop(self, *args, **kwargs):
        raise ZeroDivisionError('Templatey internal error: not implemented!')

    def remove(self, *args, **kwargs):
        raise ZeroDivisionError('Templatey internal error: not implemented!')

    def reverse(self, *args, **kwargs):
        raise ZeroDivisionError('Templatey internal error: not implemented!')

    def __imul__(self, *args, **kwargs):
        raise ZeroDivisionError('Templatey internal error: not implemented!')

    def __iadd__(self, *args, **kwargs):
        raise ZeroDivisionError('Templatey internal error: not implemented!')


type ConcreteSlotTreeNode = SlotTreeNode[ConcreteSlotTreeNode]


@dataclass(kw_only=True, slots=True)
class PendingSlotTreeNode(SlotTreeNode['PendingSlotTreeNode']):
    # Note: the str is the slot_name that the route needs to be inserted under
    insertion_slot_names: set[str] = field(default_factory=set)

    @property
    def is_insertion_point(self) -> bool:
        return bool(self.insertion_slot_names)


@dataclass(kw_only=True, slots=True)
class DynamicClassSlotTreeNode(SlotTreeNode['DynamicClassSlotTreeNode']):
    # Note: the str is the slot_name that the route needs to be inserted under
    dynamic_class_slot_names: set[str] = field(default_factory=set)

    @property
    def has_dynamic_class_slots(self) -> bool:
        return bool(self.dynamic_class_slot_names)


@dataclass(frozen=True, slots=True)
class PendingSlotTreeContainer:
    """Remember that the point here is to eventually build a lookup from
    a ``{type[template]: SlotTreeNode}``. And we're dealing with
    forward references to the ``type[template]``, meaning we don't have
    a key to use for the slot tree lookup. End of story.

    So what we're doing here instead, is constructing the slot tree as
    best as we can, and keeping track of what nodes need to be populated
    by the forward reference, once it is resolved.

    When the forward ref is resolved, we can simple copy the tree into
    all of the insertion nodes, and then store the pending slot tree
    in the slot tree lookup using the resolved template class.
    """
    pending_slot_type: ForwardRefLookupKey
    pending_root_node: PendingSlotTreeNode


@overload
def _copy_slot_tree[T: SlotTreeNode](
        src_tree: T,
        ) -> T: ...
@overload
def _copy_slot_tree[T: SlotTreeNode](
        src_tree: SlotTreeNode,
        *,
        with_node_type: type[T],
        ) -> T: ...
def _copy_slot_tree[ST: SlotTreeNode, NT: SlotTreeNode](
        src_tree: ST,
        *,
        with_node_type: type[NT] | None = None
        ) -> ST | NT:
    """This creates a copy of an existing slot tree. We use it when
    merging nested slot trees into enclosers; otherwise, we end up with
    a huge mess of "it's not clear what object holds which slot tree"
    that is very difficult to reason about. This is slightly more memory
    intensive, but... again, this is much, much easier to reason about.

    Take special note that this preserves reference cycles, which is a
    bit of a tricky thing.

    Note: if ``into_tree`` is provided, this copies inplace and returns
    the ``into_tree``. Otherwise, a new tree is created and returned.
    In both cases, we also return a lookup from
    ``{old_node.id_: copied_node}``.
    """
    copied_tree: ST | NT
    if with_node_type is None:
        node_type = type(src_tree)
    else:
        node_type = with_node_type

    copied_tree = node_type()
    copied_tree.merge_fields_only(src_tree)

    # This converts ``old_node.id_`` to the new node instance; it's how we
    # implement copying reference cycles
    transmogrified_nodes: dict[int, ST | NT] = {src_tree.id_: copied_tree}
    copy_stack: \
        list[_SlotTreeTraversalFrame[ST | NT, ST]] = [
            _SlotTreeTraversalFrame(
                next_subtree_index=0,
                existing_subtree=copied_tree,
                insertion_subtree=src_tree,
                first_encounters={})]

    while copy_stack:
        current_stack_frame = copy_stack[-1]
        if current_stack_frame.exhausted:
            copy_stack.pop()
            continue

        next_slot_route = current_stack_frame.insertion_subtree[
            current_stack_frame.next_subtree_index]
        next_slot_name, next_slot_type, next_subtree = next_slot_route
        # Do this ASAP so that we don't accidentally forget it somehow
        current_stack_frame.next_subtree_index += 1

        next_subtree_id = next_subtree.id_
        already_copied_node = transmogrified_nodes.get(next_subtree_id)
        # This could be either the first time we hit a recursive subtree,
        # or a non-recursive subtree.
        if already_copied_node is None:
            new_subtree = node_type()
            new_subtree.merge_fields_only(next_subtree)

            if next_subtree.requires_transmogrification:
                transmogrified_nodes[next_subtree_id] = new_subtree

            current_stack_frame.existing_subtree.append(
                SlotTreeRoute.new(
                    next_slot_name,
                    next_slot_type,
                    new_subtree,))
            copy_stack.append(_SlotTreeTraversalFrame(
                next_subtree_index=0,
                existing_subtree=new_subtree,
                insertion_subtree=next_subtree,
                # Note that, though this isn't correct, it also
                # doesn't matter for purely COPYING a slot tree.
                first_encounters={}))

        # We've hit a recursive subtree -- one that we've already copied --
        # which means we don't need to copy it again; instead, we just need to
        # transmogrify the reference so that the nested route refers back to
        # the original copied node.
        else:
            current_stack_frame.existing_subtree.append(
                SlotTreeRoute.new(
                    next_slot_name,
                    next_slot_type,
                    already_copied_node,))

    return copied_tree


def merge_into_slot_tree[T: SlotTreeNode](
        existing_tree_template_cls: TemplateClass,
        # Note: this is only used for determining ``is_terminus``.
        # Also note: None for pending slots. These can never be a terminus
        # anyways, so it doesn't matter.
        existing_tree_slot_type: TemplateClass | None,
        existing_tree: T,
        to_merge: T
        ) -> None:
    """This traverses the existing tree, merging in the slot_name and
    its subtrees into the correct locations in the existing slot tree,
    recursively.

    This will cull any recursion loops to their minimum possible size.
    Note that this can result in different slot tree recursion loop
    "phases", based on which of the two slot types is encountered first.

    Also note that this merges all public slot tree node attributes,
    with three exceptions:
    ++  the ``id_`` of the existing tree is always preserved
    ++  ``is_recursive`` will always be calculated based on the
        recursion loop culling
    ++  ``is_terminus`` will always be calculated based on node types
        matching the passed ``existing_tree_slot_type``.
    """
    node_type = type(existing_tree)
    transmogrified_nodes: dict[int, T] = {to_merge.id_: existing_tree}
    # Note: we want our internal culling logic to handle ``is_recursive``.
    # Also note: the existing tree usually knows whether or not it's already
    # recursive, but there's an edge case where it doesn't:
    # 1.. have a recursive loop between two templates
    # 2.. when resolving the back-ref of the second template, it finds a
    #     forward ref to itself
    # 3.. it now immediately resolves the forward ref to itself
    # In this situation, the short-circuited self-referential forward ref
    # results in is_terminus being False, which we correct here.
    existing_tree.merge_fields_only(
        to_merge, fields_to_skip={'is_recursive', 'is_terminus'})
    existing_tree.is_terminus = (
        existing_tree_template_cls is existing_tree_slot_type)

    # Counterintuitive: since we're MERGING trees, the existing_subtree is
    # actually the DESTINATION, and the insertion_subtree the source!
    merge_stack: list[_SlotTreeTraversalFrame[T, T]] = [
        _SlotTreeTraversalFrame(
            next_subtree_index=0,
            existing_subtree=existing_tree,
            insertion_subtree=to_merge,
            first_encounters={existing_tree_template_cls: existing_tree})]
    # Yes, in theory, this one specific operation of merging trees would be
    # faster if the trees were dicts instead of iterative structures. But
    # we're not optimizing for tree merging; we're optimizing for rendering!
    # And in that case, we're better off with a simple iterative structure.
    while merge_stack:
        current_stack_frame = merge_stack[-1]
        if current_stack_frame.exhausted:
            merge_stack.pop()
            continue

        existing_subtree = current_stack_frame.existing_subtree
        next_slot_route = current_stack_frame.insertion_subtree[
            current_stack_frame.next_subtree_index]
        next_slot_name, next_slot_type, next_subtree = next_slot_route
        # Do this ASAP so that we don't accidentally forget it somehow (also
        # because we want to use a continue statement in a second)
        current_stack_frame.next_subtree_index += 1

        next_subtree_id = next_subtree.id_
        already_merged_node = transmogrified_nodes.get(next_subtree_id)

        # For merging, we're going to handle the recursive subtree case first,
        # because it makes the rest of the logic cleaner
        if already_merged_node is not None:
            # Note that appending is idempotent as long as the
            # already_merged_node is the same object as any hypothetical
            # already_existing_already_merged_node that could be found there.
            # So this will only error if the slot path already exists, AND
            # it points to a different node.
            current_stack_frame.existing_subtree.append(
                SlotTreeRoute.new(
                    next_slot_name, next_slot_type, already_merged_node))

            already_merged_node.merge_fields_only(
                next_subtree,
                # Note: we want our internal culling logic to handle
                # ``is_recursive``. Otherwise, we might result in something
                # being reported as recursive, which actually isn't -- because
                # the ... phase, I guess you could say ... of the recursion
                # loop is different with a different root node.
                # Also note: we don't want to preserve the terminus value of
                # a tree we're merging in; we want the terminus to be purely
                # calculated within the merge function. This prevents bugs
                # when merging trees across slot types, which happens a lot
                # with pending slot trees.
                fields_to_skip={'is_recursive', 'is_terminus'})
            # This might seem redundant, but there are some weird edge cases
            # when merging a fully-defined tree (for example, adding in
            # self-referential recursion loops to a pending tree) that require
            # it to be set.
            # (This is because we skip it when merging the fields on the
            # existing tree; this recovers the correct state).
            already_merged_node.is_recursive = True
            continue

        # This accomplishes two things: first, it culls an extra cycle from the
        # to_merge tree that would ultimately have the same effect. Secondly,
        # it ensures correct recursion when we're merging in the pending tree
        # from a class that used us as a forward reference, since the other
        # class won't yet be resolved.
        # Note that, by definition, this won't happen the first time we
        # encounter a node of any particular type, including the type of the
        # rootwards-most node for the whole tree. This is expected! We first
        # need to encounter the slot type before we can recurse to it. **We
        # will never recurse back to the root of the slot tree!**
        if next_slot_type in current_stack_frame.first_encounters:
            # Note that we want to avoid modifying the already-encountered
            # tree. This is mostly because we copy trees willy nilly and want
            # to avoid them accidentally getting out of sync. However, it does
            # require us to be very careful to ensure that we've corrently
            # copied all needed values over prior to merging trees, if we've
            # just created the existing tree from scratch.
            next_existing_subtree = current_stack_frame.first_encounters[
                next_slot_type]
            next_existing_subtree.is_recursive = True

            next_existing_route = SlotTreeRoute.new(
                next_slot_name,
                next_slot_type,
                next_existing_subtree)
            if existing_subtree.has_route_for(next_slot_name, next_slot_type):
                existing_subtree.rewrite_route_for(
                    next_slot_name, next_slot_type, next_existing_route)
                # raise RecursionError(
                #     'Non-culled infinite recursion while merging templatey '
                #     + 'slot trees!', next_slot_name, next_slot_type)

            else:
                existing_subtree.append(next_existing_route)

            # Note that we don't need to update transmogrification for two
            # reasons: first, because we added the root node at the very
            # beginning, and second, because we -- by definition -- cannot
            # have any deeper references to this part of the destination tree,
            # because  we're culling the depthwise-rest of the source tree
            continue

        # The existing subtree -- the one we're merging INTO -- has a route for
        # this slot name and type already, so we need to merge them together
        # instead of simply copy/transmogrify/cull
        elif existing_subtree.has_route_for(next_slot_name, next_slot_type):
            next_existing_route = existing_subtree.get_route_for(
                    next_slot_name, next_slot_type)
            __, __, next_existing_subtree = next_existing_route
            next_existing_subtree.merge_fields_only(
                next_subtree,
                # Note: we want our internal culling logic to handle
                # ``is_recursive``. Otherwise, we might result in something
                # being reported as recursive, which actually isn't -- because
                # the ... phase, I guess you could say ... of the recursion
                # loop is different with a different root node.
                # Also note: we don't want to preserve the terminus value of
                # a tree we're merging in; we want the terminus to be purely
                # calculated within the merge function. This prevents bugs
                # when merging trees across slot types, which happens a lot
                # with pending slot trees.
                fields_to_skip={'is_recursive', 'is_terminus'})

        # The existing subtree doesn't have any existing routes for this, so
        # we don't need to worry about merging things together -- but we still
        # need to worry about transmogrification and culling.
        # Also note that there might be identically-named slots for different
        # slot types in the case of a union, but that will be handled on a
        # different iteration of the merge stack while loop.
        else:
            # Note that the next_subtree might have a different node type
            # than the existing tree!
            next_existing_subtree = node_type()
            next_existing_subtree.merge_fields_only(
                next_subtree,
                fields_to_skip={'is_recursive', 'is_terminus'})
            # Because we do lots and lots of merging, with offsets and
            # transforms etc, it's easiest to just reset this every time.
            # This helps prevent weird bugs with copying.
            next_existing_subtree.is_terminus = (
                # Note: what matters here is the slot type, not the template
                # class!
                next_slot_type is existing_tree_slot_type)
            next_existing_route = SlotTreeRoute.new(
                next_slot_name,
                next_slot_type,
                next_existing_subtree)
            existing_subtree.append(next_existing_route)

        if next_existing_subtree.requires_transmogrification:
            transmogrified_nodes[next_subtree_id] = next_existing_subtree

        if next_subtree:
            next_first_encounters = {**current_stack_frame.first_encounters}
            next_first_encounters.setdefault(
                next_slot_type, next_existing_subtree)
            merge_stack.append(_SlotTreeTraversalFrame(
                next_subtree_index=0,
                existing_subtree=next_existing_subtree,
                insertion_subtree=next_subtree,
                first_encounters=next_first_encounters))


def update_encloser_with_trees_from_slot(
        enclosing_cls: TemplateClass,
        enclosing_slot_tree_lookup: dict[TemplateClass, ConcreteSlotTreeNode],
        enclosing_pending_ref_lookup:
            dict[ForwardRefLookupKey, PendingSlotTreeContainer],
        enclosing_cls_ref_lookup_key: ForwardRefLookupKey,
        nested_slot_type: TemplateClass,
        nested_slot_offset: PendingSlotTreeNode,
        ) -> None:
    """This function updates the enclosing class' concrete and pending
    slot trees with the concrete and pending slot trees from the
    nested slot.

    Note that it can be called during initial signature assembly, hence
    needing explicit arguments instead of relying upon the lookups
    already being defined on the class.
    """
    nested_slot_xable = cast(type[TemplateIntersectable], nested_slot_type)
    nested_lookup = nested_slot_xable._templatey_signature._slot_tree_lookup
    nested_pending_refs = (
        nested_slot_xable._templatey_signature._pending_ref_lookup)
    # First and foremost, we can't forget to add a route for the
    # actual slot itself!
    _apply_concrete_insertions(
        enclosing_cls,
        enclosing_slot_tree_lookup,
        nested_slot_type,
        nested_lookup,
        nested_slot_type,
        # Note that this is actually supposed to be the nested slot offset
        # in both cases; the function itself gets the appropriate tree from
        # the nested_lookup
        nested_slot_offset)

    # Now the CONCRETE nested slots.
    # IMPORTANT: all of the nested concrete slots need to already
    # be merged in before starting the nested pending refs! (More
    # info below).
    for doubly_nested_slot_type in nested_lookup:
        _apply_concrete_insertions(
            enclosing_cls,
            enclosing_slot_tree_lookup,
            nested_slot_type,
            nested_lookup,
            doubly_nested_slot_type,
            # Note that this is actually supposed to be the nested slot offset
            # in both cases; the function itself gets the appropriate tree from
            # the nested_lookup
            nested_slot_offset)

    # And now, finally, the nested PENDING slots.
    # IMPORTANT: all of the nested concrete slots need to already
    # be merged in before starting the nested pending refs!
    # If we reversed the order, then we wouldn't be able to
    # guarantee that we'd already discovered every possible
    # terminus node for the nested slot type (due to potential
    # recursion loops). This could then cause us to miss out on
    # a lot of the combinatorics of the offset pending trees.
    # That being said, note that this is just a transform-and-merge
    # operation; we're not adding any additional insertion points
    # on top of what the nested slot defines. That happens later,
    # when we deal with the pending ref defs for the enclosing
    # template class (the one we're currently constructing a
    # signature for).

    # Note: we want to special-case a nested forward ref to the
    # enclosing class (ie, a recursion loop) for exactly the
    # same reason: to guarantee it's resolved before we start on
    # the other nested ones. Hence copy-and-mutate.
    unresolved_nested_forward_refs = {**nested_pending_refs}

    # Remember that we're potentially in the middle of constructing the
    # signature for a new template class. If the nested class
    # (from the slot) was depending on the class we're still
    # constructing, it hasn't yet been updated with the
    # resolved class. Therefore, instead of needing to come
    # back and fix up any recursive forward refs later, we can
    # simply do them right here, right now.
    # Also note that we'll NEVER have an existing pending tree
    # for this, because we're adding it directly and
    # immediately into the actual slot tree.
    encloser_referencing_pending_container = (
        unresolved_nested_forward_refs.pop(
            enclosing_cls_ref_lookup_key, None))
    if encloser_referencing_pending_container is not None:
        # This is effectively a special-case of pending slot tree extension
        # where we immediately apply the concrete insertions. But we still need
        # to merge the two pending trees (which handles the combinatorics --
        # remember that each insertion point for the nested class will also
        # get N insertion points from its references to the enclosing cls).
        _extend_pending_slot_tree(
            enclosing_cls=enclosing_cls,
            enclosing_slot_tree_lookup=enclosing_slot_tree_lookup,
            enclosing_pending_ref_lookup=enclosing_pending_ref_lookup,
            nested_cls=nested_slot_type,
            nested_pending_container=encloser_referencing_pending_container,
            nested_pending_ref=enclosing_cls_ref_lookup_key,)
        offset_pending_container = enclosing_pending_ref_lookup.pop(
            enclosing_cls_ref_lookup_key)
        _apply_concrete_insertions(
            enclosing_cls,
            enclosing_slot_tree_lookup,
            # These are counter-intuitive, but correct. We've divorced this
            # from the nested slot entirely; all the insertion points for it
            # now are pointed back at ourselves.
            enclosing_cls,
            enclosing_slot_tree_lookup,
            enclosing_cls,
            offset_pending_container.pending_root_node)

    # Okay, after all of that meticulous work, we can now guarantee
    # that all of the concrete classes we know about up until this
    # point have been fully resolved and incorporated into the
    # slot tree. That means we can FINALLY start merging the
    # pending items from the nested slot.
    for (
        nested_forward_ref_key, nested_pending_container
    ) in unresolved_nested_forward_refs.items():
        # Okay, so: technically, we should be checking the concrete slot tree
        # for any reference loops to the enclosing class before doing this.
        # The problem is that this can change literally every time a pending
        # ref is resolved. Therefore, we do that THEN -- and quite literally.
        # Every time a pending slot is resolved, after resolution is complete,
        # we go through every tree and merge in the recursive one.
        _extend_pending_slot_tree(
            enclosing_cls=enclosing_cls,
            enclosing_slot_tree_lookup=enclosing_slot_tree_lookup,
            enclosing_pending_ref_lookup=enclosing_pending_ref_lookup,
            nested_cls=nested_slot_type,
            nested_pending_container=nested_pending_container,
            nested_pending_ref=nested_forward_ref_key,)


def _apply_concrete_insertions(
        enclosing_cls: TemplateClass,
        enclosing_slot_tree_lookup: dict[TemplateClass, ConcreteSlotTreeNode],
        nested_slot_type: TemplateClass,
        # Note: may or may not be different from insertion_cls; see note in
        # docstring
        nested_slot_tree_lookup: dict[TemplateClass, ConcreteSlotTreeNode],
        insertion_slot_type: TemplateClass,
        insertion_tree: PendingSlotTreeNode
        ) -> None:
    """This takes an insertion tree (a pending slot tree node), applies
    a resolved, concrete insertion class at each of the insertion
    points on the insertion tree, and then merges the result into the
    slot tree lookup for the enclosing class.

    Note that this is meant to be used both with the nested class
    itself, as well as its (doubly-)nested concrete slots. In the former
    case, the slot tree lookup will match the insertion class. In the
    latter case, the insertion class will instead be the (doubly-)nested
    concrete slot.
    """
    root_after_insertion = SlotTreeNode()
    tree_to_insert = nested_slot_tree_lookup.get(
        insertion_slot_type,
        SlotTreeNode(is_terminus=True))

    # This converts ``old_node.id_`` to the new node instance; it's how we
    # implement copying reference cycles
    transmogrified_nodes: dict[int, SlotTreeNode] = {
        insertion_tree.id_: root_after_insertion}
    stack: \
        list[_SlotTreeTraversalFrame[SlotTreeNode, PendingSlotTreeNode]] = [
        _SlotTreeTraversalFrame(
            next_subtree_index=0,
            existing_subtree=root_after_insertion,
            insertion_subtree=insertion_tree,
            first_encounters={enclosing_cls: root_after_insertion})]

    while stack:
        current_stack_frame = stack[-1]
        src_subtree = current_stack_frame.insertion_subtree
        target_subtree = current_stack_frame.existing_subtree

        if current_stack_frame.exhausted:
            # This is counter-intuitive. Note that:
            # ++  the stack is for checking nested (deeper) routes only. it
            #     does not check for insertions on the current stack frame.
            # ++  the current stack frame might, though, actually have some
            #     insertions!
            # ++  we only want those insertions to be applied ONCE, and not
            #     once per subtree
            # ++  ordering of the insertion application doesn't matter
            # Therefore, we apply insertions to the current frame immediately
            # before discarding the frame for being exhausted.
            for nested_slot_type_slot_name in src_subtree.insertion_slot_names:
                target_subtree.append(
                    SlotTreeRoute.new(
                        nested_slot_type_slot_name,
                        nested_slot_type,
                        _copy_slot_tree(tree_to_insert)))

            stack.pop()
            continue

        next_slot_route = src_subtree[current_stack_frame.next_subtree_index]
        next_slot_name, next_slot_type, next_subtree = next_slot_route
        # Do this ASAP so that we don't accidentally forget it somehow
        current_stack_frame.next_subtree_index += 1

        # Note that this will still get merged into the actual full slot tree
        # for the enclosing template, which will cull any extra links in
        # recursive reference cycles, so we don't need to worry about that
        # here.
        next_subtree_id = next_subtree.id_
        transmogrified_dest_subtree = transmogrified_nodes.get(next_subtree_id)

        # This could be either the first time we hit a recursive subtree,
        # or a non-recursive subtree.
        if transmogrified_dest_subtree is None:
            dest_subtree = SlotTreeNode()
            dest_subtree.merge_fields_only(next_subtree)

            if next_subtree.requires_transmogrification:
                transmogrified_nodes[next_subtree_id] = dest_subtree

            # Note that, since we're building a new tree from scratch, we don't
            # need to worry about this already existing.
            target_subtree.append(
                SlotTreeRoute.new(
                    next_slot_name,
                    next_slot_type,
                    dest_subtree))
            next_first_encounters = {**current_stack_frame.first_encounters}
            next_first_encounters.setdefault(next_slot_type, dest_subtree)
            stack.append(_SlotTreeTraversalFrame(
                next_subtree_index=0,
                existing_subtree=dest_subtree,
                insertion_subtree=next_subtree,
                first_encounters=next_first_encounters))

        # We've hit a recursive subtree -- one that we've already copied --
        # which means we don't need to copy it again; instead, we just need to
        # transmogrify the reference so that the nested route refers back to
        # the original copied node.
        else:
            dest_subtree = transmogrified_dest_subtree
            # Note: is_recursive was already set!
            target_subtree.append(
                SlotTreeRoute.new(
                    next_slot_name,
                    next_slot_type,
                    transmogrified_dest_subtree))

    enclosing_root = enclosing_slot_tree_lookup.get(insertion_slot_type)
    if enclosing_root is None:
        enclosing_root = enclosing_slot_tree_lookup[insertion_slot_type] = (
            SlotTreeNode())

    merge_into_slot_tree(
        enclosing_cls,
        insertion_slot_type,
        enclosing_root,
        root_after_insertion)


def _extend_pending_slot_tree(
        enclosing_cls: TemplateClass,
        enclosing_slot_tree_lookup: dict[TemplateClass, ConcreteSlotTreeNode],
        enclosing_pending_ref_lookup:
            dict[ForwardRefLookupKey, PendingSlotTreeContainer],
        nested_cls: TemplateClass,
        nested_pending_container: PendingSlotTreeContainer,
        nested_pending_ref: ForwardRefLookupKey,
        ) -> None:
    """When we resolve a pending class, we do (of course) resolve the
    pending class itself. But in the process, we may also suddenly have
    a bunch of new pending refs, courtesy of the pending class.

    At first glance, you might think that we could simply dump these
    into the enclosing pending ref lookup, offset by the slot path for
    the newly-resolved class. HOWEVER, this ignores recursion, and is
    fragile in situations where classes have multiple slot names for
    a particular slot type.

    Instead, what we really need to do, is:
    ++  create a pending tree copy of the concrete slot tree for the
        newly-resolved class
    ++  at every terminus:
        ++  set ``is_terminus`` to False
        ++  insert a copy of the nested root tree
    """
    src_concrete_tree = enclosing_slot_tree_lookup[nested_cls]
    dest_copied_tree = _copy_slot_tree(
        src_concrete_tree,
        with_node_type=PendingSlotTreeNode)
    # Note that it's not an issue that this doesn't track the stack, because
    # we aren't changing the structure of the existing tree at all.
    # (We can't reuse first_encounters because it's meant for a TemplateClass)
    encountered_node_ids: set[int] = set()

    stack: \
        list[
            _SlotTreeTraversalFrame[
                PendingSlotTreeNode, ConcreteSlotTreeNode]] = [
        _SlotTreeTraversalFrame(
            next_subtree_index=0,
            existing_subtree=dest_copied_tree,
            insertion_subtree=src_concrete_tree,
            first_encounters={})]

    while stack:
        current_stack_frame = stack[-1]
        if current_stack_frame.exhausted:
            stack.pop()
            continue

        next_slot_route = current_stack_frame.insertion_subtree[
            current_stack_frame.next_subtree_index]
        next_slot_name, next_slot_type, next_subtree = next_slot_route
        # Note that we can't rely upon the indices being the same, because as
        # soon as we make an insertion, they'll drift out of sync
        target_subtree = current_stack_frame.existing_subtree.get_route_for(
            next_slot_name, next_slot_type).subtree

        # Do this ASAP so that we don't accidentally forget it somehow
        current_stack_frame.next_subtree_index += 1

        # There's never anything to do here; it means we've already done all
        # of our transforms and insertions, because it's always recursive.
        # Doesn't matter where on the tree we are. We've already fixed this
        # node, period, end of story.
        # (Don't forget that the weird doubling-back we do only applies to the
        # next EXISTING subtree, but still advances the next (insertion)
        # subtree. So we're safe in that regard.)
        if next_subtree.id_ in encountered_node_ids:
            continue
        encountered_node_ids.add(next_subtree.id_)

        if next_subtree.is_terminus:
            # As per docstring, the idea here is that we're converting every
            # terminus into an insertion point for a different (pending) slot.
            # Therefore it's no longer a terminus, since the slot is different.
            target_subtree.is_terminus = False
            merge_into_slot_tree(
                # Note: this is intentionally NOT the enclosing_cls! We're
                # working on an offset tree here, and we need to make sure that
                # we correctly report the class of the OFFSET root!
                next_slot_type,
                None,
                target_subtree,
                nested_pending_container.pending_root_node)

        # Whether or not we found a terminus, we need to check all of the
        # possibilities on the next subtree. (As a reminder, you can have a
        # terminus that still has nested nodes!)
        stack.append(_SlotTreeTraversalFrame(
            next_subtree_index=0,
            existing_subtree=target_subtree,
            insertion_subtree=next_subtree,
            # Allow the final merging to handle any culling; we don't need to
            # do it here.
            first_encounters={}))

    # Okay, the transform is complete; now we just need to insert it into
    # the actual pending slot tree for the enclosing template class.
    if nested_pending_ref in enclosing_pending_ref_lookup:
        dest_pending_tree_container = enclosing_pending_ref_lookup[
            nested_pending_ref]
    else:
        dest_pending_tree_container = enclosing_pending_ref_lookup[
            nested_pending_ref] = PendingSlotTreeContainer(
                pending_slot_type=nested_pending_ref,
                pending_root_node=PendingSlotTreeNode())

    merge_into_slot_tree(
        enclosing_cls,
        None,
        dest_pending_tree_container.pending_root_node,
        dest_copied_tree)


@dataclass(slots=True)
class _SlotTreeTraversalFrame[ET: SlotTreeNode, IT: SlotTreeNode]:
    next_subtree_index: int
    existing_subtree: ET
    insertion_subtree: IT
    first_encounters: dict[TemplateClass | None, ET]

    _insertion_subtree_len: int = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        self._insertion_subtree_len = len(self.insertion_subtree)

    @property
    def exhausted(self) -> bool:
        """Returns True if the to_merge_subtree has been exhausted, and
        there are no more subtrees to merge.
        """
        return self.next_subtree_index >= self._insertion_subtree_len


def gather_dynamic_class_slots(
        enclosing_template_cls: TemplateClass,
        enclosing_dynamic_slot_names: set[str],
        enclosing_slot_tree_lookup: dict[TemplateClass, ConcreteSlotTreeNode]
        ) -> DynamicClassSlotTreeNode:
    """This is responsible for combining all of the dynamic slots on
    the enclosing template class with all of the dynamic slots on any
    nested templates (appropriately offset from the enclosing class,
    of course).

    This will actually create the slot tree node; if you need to merge
    into an existing one (because you've resolved a forward ref), see
    ``merge_dynamic_slots``.

    Note that this can only be called **after** the enclosing slot tree
    lookup has been fully populated.
    """
    tree_root = DynamicClassSlotTreeNode(
        dynamic_class_slot_names=enclosing_dynamic_slot_names)
    # This may or may not be present in the lookup. If it is, we need to
    # handle it like any other slot, because there might be recursion loops.
    # If not... well, then we'll just skip it in the for loop.
    # (This seeming redundancy was the most practical way to do this).
    if (
        enclosing_template_cls in enclosing_slot_tree_lookup
        and enclosing_dynamic_slot_names
    ):
        merge_dynamic_class_slots(
            enclosing_template_cls,
            enclosing_slot_tree_lookup,
            tree_root,
            enclosing_template_cls,
            tree_root)

    for nested_template_cls in enclosing_slot_tree_lookup:
        # Note: more than just deduplication; it literally won't be available
        # at this point, because the signature won't be defined yet
        if nested_template_cls is not enclosing_template_cls:
            nested_template_signature = cast(
                type[TemplateIntersectable],
                nested_template_cls)._templatey_signature
            # Only merge if it actually HAS dynamic class slots! Otherwise,
            # we'd be wasting a bunch of tree traversals during render time!
            if nested_template_signature.dynamic_class_slot_names:
                merge_dynamic_class_slots(
                    enclosing_template_cls,
                    enclosing_slot_tree_lookup,
                    tree_root,
                    nested_template_cls,
                    nested_template_signature._dynamic_class_slot_tree)

    return tree_root


def merge_dynamic_class_slots(
        enclosing_template_cls: TemplateClass,
        enclosing_slot_tree_lookup: dict[TemplateClass, ConcreteSlotTreeNode],
        enclosing_dynamic_class_slot_tree: DynamicClassSlotTreeNode,
        nested_template_cls: TemplateClass,
        # This needs to be explicit, because it's not always available on the
        # nested_template_cls (yet -- if enclosing_template_cls is
        # nested_template_cls, we might still be in initial signature creation)
        nested_dynamic_class_slot_tree: DynamicClassSlotTreeNode
        ) -> None:
    """After the slot tree of a newly-resolved nested template class has
    been fully merged, call this to also update the enclosing dynamic
    slots slot tree for that particular nested_template_cls.

    Note that this modifies the tree in-place!

    This is very similar to _extend_pending_slot_tree. The general idea
    is that we search for every terminus of the existing concrete tree
    for the nested template class, and then at that terminus, we set
    the terminus to False (since it isn't used by dynamic class slots)
    and insert a copy of the nested template class' dynamic class slot
    tree.
    """
    src_concrete_tree = enclosing_slot_tree_lookup[nested_template_cls]
    dest_copied_tree = _copy_slot_tree(
        src_concrete_tree,
        with_node_type=DynamicClassSlotTreeNode)
    # Note that it's not an issue that this doesn't track the stack, because
    # we aren't changing the structure of the existing tree at all.
    # (We can't reuse first_encounters because it's meant for a TemplateClass)
    encountered_node_ids: set[int] = set()

    stack: \
        list[
            _SlotTreeTraversalFrame[
                DynamicClassSlotTreeNode, ConcreteSlotTreeNode]] = [
        _SlotTreeTraversalFrame(
            next_subtree_index=0,
            existing_subtree=dest_copied_tree,
            insertion_subtree=src_concrete_tree,
            first_encounters={})]

    while stack:
        current_stack_frame = stack[-1]
        if current_stack_frame.exhausted:
            stack.pop()
            continue

        next_slot_route = current_stack_frame.insertion_subtree[
            current_stack_frame.next_subtree_index]
        next_slot_name, next_slot_type, next_subtree = next_slot_route
        # Note that we can't rely upon the indices being the same, because as
        # soon as we make an insertion, they'll drift out of sync
        target_subtree = current_stack_frame.existing_subtree.get_route_for(
            next_slot_name, next_slot_type).subtree

        # Do this ASAP so that we don't accidentally forget it somehow
        current_stack_frame.next_subtree_index += 1

        # There's never anything to do here; it means we've already done all
        # of our transforms and insertions, because it's always recursive.
        # Doesn't matter where on the tree we are. We've already fixed this
        # node, period, end of story.
        # (Don't forget that the weird doubling-back we do only applies to the
        # next EXISTING subtree, but still advances the next (insertion)
        # subtree. So we're safe in that regard.)
        if next_subtree.id_ in encountered_node_ids:
            continue
        encountered_node_ids.add(next_subtree.id_)

        if next_subtree.is_terminus:
            # As per docstring, the idea here is that we're converting every
            # terminus into an insertion point for a different (pending) slot.
            # Therefore it's no longer a terminus, since the slot is different.
            target_subtree.is_terminus = False
            merge_into_slot_tree(
                # Note: this is intentionally NOT the enclosing_cls! We're
                # working on an offset tree here, and we need to make sure that
                # we correctly report the class of the OFFSET root!
                next_slot_type,
                None,
                target_subtree,
                nested_dynamic_class_slot_tree)

        # Whether or not we found a terminus, we need to check all of the
        # possibilities on the next subtree. (As a reminder, you can have a
        # terminus that still has nested nodes!)
        stack.append(_SlotTreeTraversalFrame(
            next_subtree_index=0,
            existing_subtree=target_subtree,
            insertion_subtree=next_subtree,
            # Allow the final merging to handle any culling; we don't need to
            # do it here.
            first_encounters={}))

    merge_into_slot_tree(
        enclosing_template_cls,
        None,
        enclosing_dynamic_class_slot_tree,
        dest_copied_tree)


@dataclass(slots=True)
class _DynaClsExtractorFrame:
    active_instance: TemplateParamsInstance
    active_subtree: DynamicClassSlotTreeNode
    target_subtree_index: int
    target_instance_index: int
    target_instances_count: int = field(kw_only=True, default=0)
    target_instances: Sequence[TemplateParamsInstance] = field(
        kw_only=True, init=False)

    _active_subtree_len: int = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        self._active_subtree_len = len(self.active_subtree)

    @property
    def exhausted(self) -> bool:
        return self.target_subtree_index >= self._active_subtree_len


def extract_dynamic_template_classes(
        root_template_instance: TemplateParamsInstance,
        dynamic_class_slot_tree: DynamicClassSlotTreeNode
        ) -> set[TemplateClass]:
    """Given a root template instance and its associated dynamic class
    slot tree, walks the tree and extracts all template classes for
    slots defined as having a dynamic class.

    This does no culling based on whether or not the class was already
    loaded; it simply constructs a set of all encountered dynamic
    template classes.
    """
    dynamic_template_classes: set[TemplateClass] = set()
    stack: list[_DynaClsExtractorFrame] = [
        _DynaClsExtractorFrame(
            active_instance=root_template_instance,
            active_subtree=dynamic_class_slot_tree,
            target_subtree_index=0,
            target_instance_index=0)]

    while stack:
        frame = stack[-1]
        active_instance = frame.active_instance
        active_subtree = frame.active_subtree

        if frame.exhausted:
            # Yes, we only want to do this when the frame is exhausted. This
            # way, we only perform the check once per node instead of once
            # per subtree!
            for dynamic_slot_name in active_subtree.dynamic_class_slot_names:
                dynamic_template_classes.update(
                    type(slot_instance)
                    for slot_instance
                    in getattr(active_instance, dynamic_slot_name))

            stack.pop()
            continue

        slot_route = frame.active_subtree[frame.target_subtree_index]
        slot_name, slot_type, subtree = slot_route
        target_instance_index = frame.target_instance_index

        # This is, in a way, a nested stack, but we're maintaining
        # the stack state within the _DynaClsExtractorFrame.
        # At any rate, we use the zero-index iteration of the loop to
        # memoize some values on the stack frame.
        if target_instance_index == 0:
            target_instances = getattr(active_instance, slot_name)
            target_instances_count = len(target_instances)

            # Check in advance if there are no target instances at all,
            # and if so, skip the whole thing. This isn't just for
            # performance; the processing logic depends on it.
            if target_instances_count > 0:
                frame.target_instances_count = target_instances_count
                frame.target_instances = target_instances
            else:
                # Note: this is critical! Otherwise we'll infinitely loop.
                frame.target_subtree_index += 1
                continue

        else:
            target_instances_count = frame.target_instances_count
            # We've exhausted the target instances; reset the state for
            # the next slot tree route and then continue.
            if frame.target_instance_index >= target_instances_count:
                # Note: we're deliberately skipping the target instances
                # themselves, because it'll just get overwritten the next
                # time around, so we can save ourselves an operation.
                frame.target_instances_count = 0
                frame.target_instance_index = 0
                # Note: this is critical! Otherwise we'll infinitely loop.
                frame.target_subtree_index += 1
                continue

            # We still have some instances to target; normalize the state
            # so that we can operate on them.
            target_instances = frame.target_instances

        # Okay, status check: we have our stack frame state configured
        # correctly, and we have target instances to check.
        instance_to_check = target_instances[target_instance_index]
        # Note: exact match here; not subclassing! Subclassing breaks too
        # many things, so we don't support it.
        if type(instance_to_check) is slot_type:
            stack.append(_DynaClsExtractorFrame(
                active_instance=instance_to_check,
                active_subtree=subtree,
                target_subtree_index=0,
                target_instance_index=0,))

        frame.target_instance_index += 1

    return dynamic_template_classes
