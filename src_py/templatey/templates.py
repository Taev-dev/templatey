from __future__ import annotations

import functools
import inspect
import itertools
import logging
import typing
from collections import defaultdict
from collections import deque
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import Field
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from textwrap import dedent
from types import EllipsisType
from typing import Annotated
from typing import Any
from typing import ClassVar
from typing import Literal
from typing import Protocol
from typing import cast
from typing import dataclass_transform
from typing import runtime_checkable

try:
    from typing import TypeIs
except ImportError:
    from typing_extensions import TypeIs

from templatey._annotations import InterfaceAnnotation
from templatey._annotations import InterfaceAnnotationFlavor
from templatey.interpolators import NamedInterpolator
from templatey.parser import InterpolatedVariable
from templatey.parser import NestedContentReference
from templatey.parser import NestedVariableReference
from templatey.parser import ParsedTemplateResource

if typing.TYPE_CHECKING:
    from _typeshed import DataclassInstance
else:
    DataclassInstance = object

# Technically this should be an intersection type with both the
# _TemplateIntersectable from templates and the DataclassInstance returned by
# the dataclass transform. Unfortunately, type intersections don't yet exist in
# python, so we have to resort to this (overly broad) type
TemplateParamsInstance = DataclassInstance
logger = logging.getLogger(__name__)


type TemplateClass = type[TemplateParamsInstance]
# Technically, these should use the TemplateIntersectable from templates.py,
# but since we can't define type intersections yet...
type Slot[T: TemplateParamsInstance] = Annotated[
    Sequence[T] | EllipsisType,
    InterfaceAnnotation(InterfaceAnnotationFlavor.SLOT)]
type Var[T] = Annotated[
    T | EllipsisType,
    InterfaceAnnotation(InterfaceAnnotationFlavor.VARIABLE)]
type Content[T] = Annotated[
    T,
    InterfaceAnnotation(InterfaceAnnotationFlavor.CONTENT)]


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


class VariableEscaper(Protocol):

    def __call__(self, value: str) -> str:
        """Variable escaper functions accept a single positional
        argument: the value of the variable to escape. It then does any
        required escaping and returns the final string.
        """
        ...


class ContentVerifier(Protocol):

    def __call__(self, value: str) -> Literal[True]:
        """Content verifier functions accept a single positional
        argument: the value of the content to verify. It does any
        verification, and then returns True if the content was okay,
        or raises BlockedContentValue if the content was not acceptable.

        Note that we raise instead of trying to escape for two reasons:
        1.. We don't really know what to replace it with. This is also
            true with variables, but:
        2.. We expect that content is coming from -- if not trusted,
            then at least authoritative -- sources, and therefore, we
            should fail loudly, because it gives the author a chance to
            correct the problem before it becomes user-facing.
        """
        ...


@dataclass_transform()
def template[T: type](  # noqa: PLR0913
        config: TemplateConfig,
        template_resource_locator: object,
        /, *,
        init: bool = True,
        repr: bool = True,  # noqa: A002
        eq: bool = True,
        order: bool = False,
        unsafe_hash: bool = False,
        frozen: bool = False,
        match_args: bool = True,
        kw_only: bool = False,
        slots: bool = False,
        weakref_slot: bool = False
        ) -> Callable[[T], T]:
    """This both transforms the decorated class into a dataclass, and
    declares it as a templatey template.
    """
    return functools.partial(
        make_template_definition,
        dataclass_kwargs={
            'init': init,
            'repr': repr,
            'eq': eq,
            'order': order,
            'unsafe_hash': unsafe_hash,
            'frozen': frozen,
            'match_args': match_args,
            'kw_only': kw_only,
            'slots': slots,
            'weakref_slot': weakref_slot
        },
        template_resource_locator=template_resource_locator,
        template_config=config)


@dataclass(frozen=True)
class TemplateConfig[T: type, L: object]:
    interpolator: NamedInterpolator
    variable_escaper: VariableEscaper
    content_verifier: ContentVerifier


@dataclass(slots=True, frozen=True)
class TemplateProvenanceNode:
    """TemplateProvenanceNode instances are unique to the exact location
    on the exact render tree of a particular template instance. If the
    template instance gets reused within the same render tree, it will
    have multiple provenance nodes. And each different slot in a parent
    will have a separate provenance node, potentially with different
    namespace overrides.

    This is used both during function execution (to calculate the
    concrete value of any parameters), and also while flattening the
    render tree.

    Also note that the root template, **along with any templates
    injected into the render by environment functions,** will have an
    empty list of provenance nodes.

    Note that, since overrides from the parent can come exclusively from
    the template body -- and are therefore shared across all children of
    the same slot -- they don't get stored within the provenance, since
    we'd require access to the template bodies, which we don't yet have.
    """
    parent_slot_key: str
    parent_slot_index: int
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
        parent_param_name = name
        parent_slot_key = current_provenance_node.parent_slot_key
        for parent in reversed(self[0:-1]):
            template_class = type(parent.instance)
            parent_template = template_preload[template_class]
            parent_overrides = parent_template.slots[parent_slot_key].params

            if parent_param_name in parent_overrides:
                value = parent_overrides[parent_param_name]

                if isinstance(value, NestedContentReference):
                    parent_slot_key = parent.parent_slot_key
                    parent_param_name = value.name
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
        value = getattr(current_provenance_node, name, ...)
        parent_param_name = name
        parent_slot_key = current_provenance_node.parent_slot_key
        for parent in reversed(self[0:-1]):
            template_class = type(parent.instance)
            parent_template = template_preload[template_class]
            parent_overrides = parent_template.slots[parent_slot_key].params

            if parent_param_name in parent_overrides:
                value = parent_overrides[parent_param_name]

                if isinstance(value, NestedVariableReference):
                    parent_slot_key = parent.parent_slot_key
                    parent_param_name = value.name
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


type _SlotTreeNode = tuple[_SlotTreeRoute, ...]
type _SlotTreeRoute = tuple[str, _SlotTreeNode]
type TemplateInstanceID = int
type GroupedTemplateInvocations = dict[TemplateClass, list[TemplateProvenance]]
type TemplateLookupByID = dict[TemplateInstanceID, TemplateParamsInstance]


@dataclass(slots=True)
class TemplateSignature:
    """This class stores the processed interface based on the params.
    It gets used to compare with the TemplateParse to make sure that
    the two can be used together.
    """
    slots: dict[str, type[TemplateParamsInstance]]
    slot_names: set[str]
    vars_: dict[str, type]
    var_names: set[str]
    content: dict[str, type]
    content_names: set[str]
    _slot_tree_lookup: dict[
        type[TemplateParamsInstance], _SlotTreeNode] = field(
            default_factory=dict)
    # All of these are set during __post_init__
    get_all_vars: Callable[
        [TemplateParamsInstance], dict[str, object]] = None  # type: ignore
    get_var: Callable[
        [TemplateParamsInstance, str], object] = None #  type: ignore
    get_all_content: Callable[
        [TemplateParamsInstance], dict[str, object]] = None  # type: ignore
    get_content: Callable[
        [TemplateParamsInstance, str], object] = None #  type: ignore

    def __post_init__(self):
        tree_wip: dict[type[TemplateParamsInstance], list[_SlotTreeRoute]]
        tree_wip = defaultdict(list)
        for parent_slot_name, parent_slot_type in self.slots.items():
            slot_xable = cast(type[TemplateIntersectable], parent_slot_type)
            child_lookup = slot_xable._templatey_signature._slot_tree_lookup
            for child_slot_type, child_slot_tree in child_lookup.items():
                tree_wip[child_slot_type].append(
                    (parent_slot_name, child_slot_tree))

            # Note that the empty tuple here denotes that it doesn't have any
            # children **for the current node.** That doesn't mean that the
            # child tree doesn't have any other slots of the same type (hence
            # using append), but we're mapping ALL of the nodes, and NOT just
            # the leaves.
            tree_wip[parent_slot_type].append((parent_slot_name, ()))

        for slot_key, route_list in tree_wip.items():
            self._slot_tree_lookup[slot_key] = tuple(route_list)

        def all_vars_getter(
                instance: TemplateParamsInstance,
                _var_names=self.var_names
                ) -> dict[str, object]:
            return {name: getattr(instance, name) for name in _var_names}

        def single_var_getter(
                instance: TemplateParamsInstance,
                name: str,
                _var_names=self.var_names
                ) -> dict[str, object]:
            if name not in _var_names:
                raise KeyError('Not a variable!', name)
            return getattr(instance, name)

        def all_content_getter(
                instance: TemplateParamsInstance,
                _content_names=self.content_names
                ) -> dict[str, object]:
            return {name: getattr(instance, name) for name in _content_names}

        def single_content_getter(
                instance: TemplateParamsInstance,
                name: str,
                _content_names=self.content_names
                ) -> dict[str, object]:
            if name not in _content_names:
                raise KeyError('Not a content!', name)
            return getattr(instance, name)

        self.get_all_vars = all_vars_getter
        self.get_var = single_var_getter
        self.get_all_content = all_content_getter
        self.get_content = single_content_getter

    def apply_slot_tree(
            self,
            root_template_instance: TemplateParamsInstance
            ) -> tuple[GroupedTemplateInvocations, TemplateLookupByID]:
        """
        """
        template_lookup: TemplateLookupByID = {}
        template_invocations: GroupedTemplateInvocations = defaultdict(list)
        # The actual logic here can be pretty confusing. The comments should
        # help, but the test cases -- both of building and using the slot
        # tree -- are extremely helpful when following through the logic. If
        # you need to understand this, I recommend rubber-ducking it with a
        # few of those test cases.
        current_subtree: _SlotTreeNode
        slot_class_invocations: list[TemplateProvenance]

        child_slot_name: str | None
        child_routes: tuple[_SlotTreeRoute, ...]
        sibling_routes: list[_SlotTreeRoute]
        # Backlog is a stack of stacks. The outer stack corresponds to the
        # depth of the slot tree; the inner stack corresponds to siblings at
        # a given depth level.
        # The purpose is to keep track of the current search path, so that we
        # can back out and try other ROUTE siblings (attribute keys! not
        # instances!) after exhausting a tree branch.
        backlog: list[list[_SlotTreeRoute]]
        # Visited is a single stack. The outer list (the stack) corresponds to
        # the depth of the slot tree; the inner list (not a stack) corresponds
        # to all discovered instances at that depth level.
        # The purpose is to keep track of all instances at a particular tree
        # depth, so that when we pop depth levels off the backlog, we can
        # recover all of the previousls-discovered instances (templates! not
        # attribute key strings!) from the shallower depth level
        visited: list[list[TemplateProvenanceNode]]

        instances_at_child_slot_name: list[Sequence[TemplateParamsInstance]]

        for slot_class, current_subtree in self._slot_tree_lookup.items():
            slot_class_invocations = []
            # We need to initialize the search first, based on the passed
            # parameters. We'll then mutate the backlog and visited structures
            # in-place while performing our search.
            (child_slot_name, child_routes), *sibling_routes = current_subtree
            backlog = [sibling_routes]

            # Note that:
            # 1.. the final provenance needs to be a tuple (subclass),
            #     requiring this to be copied anyways
            # 2.. the root node is never supposed to show up in the provenance
            # 3.. therefore, we can (at basically no cost) simply index the
            #     tuple from visited[1:]
            # 4.. therefore, the meaningless slot key and slot index are a
            #     perfectly reasonable way to bootstrap this in a typesafe way
            #     that avoids needless "is not None" checks
            visited = [[TemplateProvenanceNode(
                parent_slot_key='',
                parent_slot_index=0,
                instance_id=id(root_template_instance),
                instance=root_template_instance)]]

            while child_slot_name is not None:
                print(child_slot_name)
                print(visited[-1])
                # [*root_instance.foo]
                slot_counter = itertools.count()
                print(visited[-1][0])
                print('---')
                print(getattr(visited[-1][0].instance, child_slot_name))

                # This is actually a list of a list after this call

                instances_at_child_slot_name = [
                    getattr(template_provenance.instance, child_slot_name)
                    for template_provenance in visited[-1]]
                provenances_at_child_slot_name = (
                    TemplateProvenanceNode(
                        parent_slot_key=child_slot_name,
                        parent_slot_index=next(slot_counter),
                        instance_id=id(child_instance),
                        instance=child_instance)
                    for child_instance in instances_at_child_slot_name)

                print('#####')
                print(provenances_at_child_slot_name)

                # This means that the slot tree has more depth to it that we
                # still need to explore, so we'll be pushing to both stacks.
                if child_routes:
                    # ('bar', (...), (...)), []
                    (child_slot_name, child_routes), *sibling_routes = (
                        child_routes)

                    visited.append(list(provenances_at_child_slot_name))
                    backlog.append(sibling_routes)

                # This means that we've reached the deepest point of this
                # particular slot tree branch. That means we found an instance,
                # and need to add it to the found instances, before continuing
                # on with either a sibling, or by popping out one depth level.
                else:
                    # Other options tried for appending the current provenance:
                    # ++  a custom def _append(iter, val): yield from iter;
                    #     yield val
                    # ++  nested generator comprehensions
                    # ++  itertools.chain
                    # This was faster than any of the competitors, and probably
                    # also the easiest to understand.
                    parent_provenance_nodes = tuple(
                        provenances[-1] for provenances in visited[1:])
                    slot_class_invocations.extend(
                        TemplateProvenance(
                            (*parent_provenance_nodes, child_provenance))
                        for child_provenance
                        in provenances_at_child_slot_name)
                    child_slot_name = None

                    # The while here is in case we need to back out multiple
                    # depth levels
                    while child_slot_name is None and backlog:
                        print('#####')
                        print(backlog[-1])
                        # The current depth level has siblings, so we need to
                        # search down their branches
                        if backlog[-1]:
                            child_slot_name, child_routes = backlog[-1].pop()
                        # There are no more siblings at the current branch
                        # depth. That means we need to pop both stacks to
                        # decrease depth by one, and then continue the search
                        # at that level.
                        else:
                            backlog.pop()
                            visited.pop()

            template_invocations[slot_class].extend(slot_class_invocations)

        # Last but not least, don't forget the root!
        template_invocations[type(root_template_instance)].append(
            TemplateProvenance((
                TemplateProvenanceNode(
                    parent_slot_key='',
                    parent_slot_index=-1,
                    instance_id=id(root_template_instance),
                    instance=root_template_instance),)),)
        template_lookup[id(root_template_instance)] = root_template_instance

        return template_invocations, template_lookup


@dataclass_transform()
def make_template_definition[T: type](
        cls: T,
        *,
        dataclass_kwargs: dict[str, bool],
        # Note: needs to be understandable by template loader
        template_resource_locator: object,
        template_config: TemplateConfig
        ) -> T:
    """Programmatically creates a template definition. Converts the
    requested class into a dataclass, passing along ``dataclass_kwargs``
    to the dataclass constructor. Then performs some templatey-specific
    bookkeeping. Returns the resulting dataclass.
    """
    cls = dataclass(**dataclass_kwargs)(cls)
    cls._templatey_config = template_config
    cls._templatey_resource_locator = template_resource_locator

    # We're prioritizing the typical case here, where the templates are defined
    # at the module toplevel, and therefore accessible within the module
    # globals. However, if the template is defined within a closure, we might
    # need to walk up the stack until we find a caller that isn't within this
    # file, and then grab its locals.
    try:
        template_type_hints = typing.get_type_hints(cls)
    except NameError as exc:
        template_type_hints = None
        maybe_locals = _extract_template_class_locals()
        if maybe_locals is not None:
            try:
                template_type_hints = typing.get_type_hints(
                    cls, localns=maybe_locals)

            # We'll just revert to the parent exception in this case
            except NameError:
                pass

        if template_type_hints is None:
            exc.add_note(
                dedent('''\
                This NameError was raised while trying to get the type hints
                assigned to a class decorated with @templatey.template.
                This typically means you were creating a template within a
                closure, and we were unable to infer the locals via
                inspect.currentframe (probably because your current python
                platform doesn't support it). Alternatively, this may be the
                result of attempting to use a forward reference within the
                type hint; note that, though the type will resolve correctly,
                the actual class still isn't defined at this point, preventing
                type hint resolution. In that case, simply make sure to declare
                any child slot templates before their parents reference them.
                '''
                ))
            raise exc

    slots = {}
    slot_names = set()
    vars_ = {}
    var_names = set()
    content = {}
    content_names = set()
    for template_field in fields(cls):
        field_classification = _classify_interface_field_flavor(
            template_type_hints, template_field)

        # Note: it's not entirely clear to me that this restriction makes
        # sense; I could potentially see MAYBE there being some kind of
        # template function that could access other attributes from the
        # dataclass? But also, maybe those should be vars? Again, unclear.
        if field_classification is None:
            raise TypeError(
                'Template parameter definitions may only contain variables, '
                + 'slots, and content!')

        else:
            field_flavor, wrapped_type = field_classification

            # A little awkward to effectively just repeat the comparison we did
            # when classifying, but that makes testing easier and control flow
            # clearer
            if field_flavor is InterfaceAnnotationFlavor.VARIABLE:
                dest_lookup = vars_
                dest_names = var_names
            elif field_flavor is InterfaceAnnotationFlavor.SLOT:
                dest_lookup = slots
                dest_names = slot_names
            else:
                dest_lookup = content
                dest_names = content_names

            dest_lookup[template_field.name] = wrapped_type
            dest_names.add(template_field.name)

    cls._templatey_signature = TemplateSignature(
        slots=slots,
        slot_names=slot_names,
        vars_=vars_,
        var_names=var_names,
        content=content,
        content_names=content_names)
    return cls


def _extract_template_class_locals() -> dict[str, Any] | None:
    upstack_frame = inspect.currentframe()
    if upstack_frame is None:
        return None
    else:
        this_module = upstack_module = inspect.getmodule(
            _extract_template_class_locals)
        while upstack_module is this_module:
            if upstack_frame is None:
                return None

            upstack_frame = upstack_frame.f_back
            upstack_module = inspect.getmodule(upstack_frame)

    if upstack_frame is not None:
        return upstack_frame.f_locals


def _classify_interface_field_flavor(
        parent_class_type_hints: dict[str, Any],
        template_field: Field
        ) -> tuple[InterfaceAnnotationFlavor, type] | None:
    """For a dataclass field, determines whether it was declared as a
    var, slot, or content.

    If none of the above, returns None.
    """
    # Note that dataclasses don't include the actual type (just a string)
    # when in __future__ mode, so we need to get them from the parent class
    # by calling get_type_hints() on it
    resolved_field_type = parent_class_type_hints[template_field.name]
    anno_origin = typing.get_origin(resolved_field_type)
    if anno_origin is Var:
        nested_type, = typing.get_args(resolved_field_type)
        return InterfaceAnnotationFlavor.VARIABLE, nested_type
    elif anno_origin is Slot:
        nested_type, = typing.get_args(resolved_field_type)
        return InterfaceAnnotationFlavor.SLOT, nested_type
    elif anno_origin is Content:
        nested_type, = typing.get_args(resolved_field_type)
        return InterfaceAnnotationFlavor.CONTENT, nested_type
    else:
        return None


@dataclass(frozen=True)
class InjectedValue:
    """This is used by template functions to indicate that a value is
    being injected into the template by the function. It can indicate
    whether verification, escaping, or both should be applied to the
    value after conversion to a string.
    """
    value: object
    format_spec: str | None
    conversion: str | None

    use_content_verifier: bool = False
    use_variable_escaper: bool = True


@runtime_checkable
class ComplexContent(Protocol):
    """Sometimes content isn't as simple as a ``string``. For example,
    content might include variable interpolations. Or you might need
    to modify the content slightly based on the variables -- for
    example, to get subject/verb alignment based on a number, gender
    alignment based on a pronoun, or whatever. ComplexContent gives
    you an escape hatch to do this: simply pass a ComplexContent
    instance as a value instead of a string.
    """
    TEMPLATEY_CONTENT: ClassVar[Literal[True]] = True

    def flatten(
            self,
            unescaped_vars_context: dict[str, object]
            ) -> Iterable[str | InterpolatedVariable]:
        """Implement this for any instance of complex content. **Note
        that you should never perform the variable interpolation
        yourself.** Instead, you should do whatever content modification
        you need based on the variables, yielding back an
        InterpolatedVariable placeholder in place of the value. This
        lets templatey manage variable escaping, etc.
        """
        ...
