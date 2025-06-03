from __future__ import annotations

import functools
import inspect
import itertools
import logging
import typing
from collections import defaultdict
from collections import deque
from collections.abc import Callable
from collections.abc import Collection
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import Field
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from textwrap import dedent
from types import EllipsisType
from types import UnionType
from typing import Annotated
from typing import Any
from typing import ClassVar
from typing import Literal
from typing import Protocol
from typing import cast
from typing import dataclass_transform
from typing import runtime_checkable

try:
    from typing import TypeIs  # type: ignore
except ImportError:
    from typing_extensions import TypeIs

from docnote import ClcNote

from templatey._annotations import InterfaceAnnotation
from templatey._annotations import InterfaceAnnotationFlavor
from templatey.interpolators import NamedInterpolator
from templatey.parser import InterpolatedFunctionCall
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
    interpolator: Annotated[
        NamedInterpolator,
        ClcNote(
            '''The interpolator determines what characters are used for
            performing interpolations within the template. They can be
            escaped by repeating them, for example ``{{}}`` would be
            a literal ``{}`` with a curly braces interpolator.
            ''')]
    variable_escaper: Annotated[
        VariableEscaper,
        ClcNote(
            '''Variables are always escaped. The variable escaper is
            the callable responsible for performing that escaping. If you
            don't need escaping, there are noop escapers within the prebaked
            template configs that you can use for convenience.
            ''')]
    content_verifier: Annotated[
        ContentVerifier,
        ClcNote(
            '''Content isn't escaped, but it ^^is^^ verified. Content
            verification is a simple process that either succeeds or fails;
            it allows, for example, to allowlist certain HTML tags.
            ''')]


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
        value = getattr(current_provenance_node.instance, name, ...)
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
                'No value found for variable with name at slot!',
                self[-1].instance, name)

        return value


type _SlotTreeNode = tuple[_SlotTreeRoute, ...]
type _SlotTreeRoute = tuple[str, _SlotTreeNode]
type TemplateInstanceID = int
type GroupedTemplateInvocations = dict[TemplateClass, list[TemplateProvenance]]
type TemplateLookupByID = dict[TemplateInstanceID, TemplateParamsInstance]
# Note that there's no need for an abstract version of this, at least not right
# now, because in order to calculate it, we'd need to know the template body,
# which doesn't happen until we already know the template instance, which means
# we can skip ahead to the concrete version.
type EnvFuncInvocation = tuple[TemplateProvenance, InterpolatedFunctionCall]


@dataclass(slots=True)
class TemplateSignature:
    """This class stores the processed interface based on the params.
    It gets used to compare with the TemplateParse to make sure that
    the two can be used together.

    Not meant to be created directly; instead, you should use the
    TemplateSignature.new() convenience method.
    """
    slots: Mapping[str, type[TemplateParamsInstance] | UnionType]
    slot_names: frozenset[str]
    vars_: dict[str, type]
    var_names: frozenset[str]
    content: dict[str, type]
    content_names: frozenset[str]
    included_template_classes: frozenset[type[TemplateParamsInstance]]

    # Note that this contains all included types, not just the ones on the
    # outermost layer that are associated with the signature
    _slot_tree_lookup: dict[type[TemplateParamsInstance], _SlotTreeNode]

    @classmethod
    def new(
            cls,
            slots: dict[str, type[TemplateParamsInstance] | UnionType],
            vars_: dict[str, type],
            content: dict[str, type]
            ) -> TemplateSignature:
        """Create a new TemplateSignature based on the gathered slots,
        vars, and content. This does all of the convenience calculations
        needed to populate the semi-redundant fields.
        """
        slot_names = frozenset(slots)
        var_names = frozenset(vars_)
        content_names = frozenset(content)

        slot_tree_lookup = {}
        tree_wip: dict[type[TemplateParamsInstance], list[_SlotTreeRoute]]
        tree_wip = defaultdict(list)
        for parent_slot_name, parent_slot_annotation in slots.items():
            parent_slot_types: Collection[type[TemplateParamsInstance]]
            if isinstance(parent_slot_annotation, UnionType):
                parent_slot_types = parent_slot_annotation.__args__
            else:
                parent_slot_types = (parent_slot_annotation,)

            for parent_slot_type in parent_slot_types:
                slot_xable = cast(
                    type[TemplateIntersectable], parent_slot_type)
                child_lookup = (
                    slot_xable._templatey_signature._slot_tree_lookup)
                for child_slot_type, child_slot_tree in child_lookup.items():
                    tree_wip[child_slot_type].append(
                        (parent_slot_name, child_slot_tree))

                # Note that the empty tuple here denotes that it doesn't have
                # any children **for the current node.** That doesn't mean that
                # the child tree doesn't have any other slots of the same type
                # (hence using append), but we're mapping ALL of the nodes, and
                # NOT just the leaves.
                tree_wip[parent_slot_type].append((parent_slot_name, ()))

        for slot_key, route_list in tree_wip.items():
            slot_tree_lookup[slot_key] = tuple(route_list)

        return cls(
            slots=slots,
            slot_names=slot_names,
            vars_=vars_,
            var_names=var_names,
            content=content,
            content_names=content_names,
            _slot_tree_lookup=slot_tree_lookup,
            included_template_classes=frozenset(slot_tree_lookup))

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

        # Things to keep in mind when reading the following code:
        # ++  it may be easiest to step through using an example, perhaps from
        #     the test suite.
        # ++  parent template classes can have multiple slots with the same
        #     child template class. We call these "parallel slots" in this
        #     function; they're places where the slot search tree needs to
        #     split into multiple branches
        # ++  the combinatorics here can be confusing, because we can have both
        #     multiple instances in each slot AND multiple slots for each
        #     template class
        # ++  multiple instances per slot branch the instance search tree, but
        #     multiple slots per template class branch the slot search tree.
        #     However, we need to exhaust both search trees when finding all
        #     relevant provenances. Therefore, we have to periodically refresh
        #     one of the search trees, whenever we step to a sibling on the
        #     other tree

        # The goal of the parallel slot backlog is to keep track of which other
        # SLOTS (attributes! not instances!) at a particular instance are ALSO
        # on the search path, and therefore need to be returned to after the
        # current branch has been exhausted.
        # The parallel slot backlog is a stack of stacks. The outer stack
        # corresponds to the depth in the slot tree. The inner stack
        # contains all remaining slots to search for a particular depth.
        parallel_slot_backlog_stack: list[list[_SlotTreeRoute]]

        # The goal of the instance backlog stack is to keep track of all the
        # INSTANCES (not slots / attributes!) that are also on the search path,
        # and therefore need to be returned to after the current branch has
        # been exhausted. Its deepest level gets refreshed from the instance
        # history stack every time we move on to a new parallel slot.
        # Similar (but different) to the parallel slot backlog, the instance
        # backlog is a stack of **queues** (it's important to preserve order,
        # since slot members are by definition ordered). The outer stack
        # again corresponds to the depth in the slot tree, and the inner queue
        # contains all of the remaining instances to search for a particular
        # depth.
        instance_backlog_stack: list[deque[TemplateProvenanceNode]]

        # The instance history stack is similar to the backlog stack; however,
        # it is not mutated on a particular level. Use it to "refresh" the
        # instance backlog stack for parallel slots.
        instance_history_stack: list[tuple[TemplateProvenanceNode, ...]]

        # These are all used per-iteration, and don't keep state across
        # iterations.
        child_slot_name: str
        child_slot_routes: tuple[_SlotTreeRoute, ...]
        child_instances: Sequence[TemplateParamsInstance]

        # This is used in multiple loop iterations, plus at the end to add any
        # function calls for the root instance.
        root_provenance_node = TemplateProvenanceNode(
            parent_slot_key='',
            parent_slot_index=-1,
            instance_id=id(root_template_instance),
            instance=root_template_instance)

        # Keep in mind that the slot tree contains all included slot classes
        # (recursively), not just the ones at the root_template_instance.
        # Our goal here is:
        # 1.. find all template classes with abstract function calls
        # 2.. build provenances for all invocations of those template classes
        # 3.. combine (product) those provenances with all of the abstract
        #     function calls at that template class
        for template_class, root_nodes in self._slot_tree_lookup.items():
            abstract_calls = template_preload[template_class].function_calls

            # Constructing a provenance is relatively expensive, so we only
            # want to do it if we actually have some function calls within the
            # template
            if abstract_calls:
                provenances: list[TemplateProvenance] = []
                parallel_slot_backlog_stack = [list(root_nodes)]
                instance_history_stack = [(root_provenance_node,)]
                instance_backlog_stack = [deque(instance_history_stack[0])]

                # Our overall strategy here is to let the instance stack be
                # the primary driver. Only mutate other stacks when we're done
                # with a particular level in the instance backlog!
                while instance_backlog_stack:
                    # If there's nothing left on the current level of the
                    # instance backlog, there are a couple options.
                    # ++  We may have exhausted a particular parallel path
                    #     from the current level, but there are more left. We
                    #     need to refresh the list of instances and continue.
                    # ++  We may have exhausted all subtrees of the current
                    #     level. In that case, we need to back out a level and
                    #     continue looking for parallels, one level up.
                    if not instance_backlog_stack[-1]:
                        # Note that by checking for >1, we don't allocate a
                        # bunch of instance backlog children for nothing.
                        if len(parallel_slot_backlog_stack[-1]) > 1:
                            parallel_slot_backlog_stack[-1].pop()
                            instance_backlog_stack[-1].extend(
                                instance_history_stack[-1])

                        else:
                            parallel_slot_backlog_stack.pop()
                            instance_backlog_stack.pop()
                            instance_history_stack.pop()

                    # There are one or more remaining parallel paths from the
                    # current instances that lead to the target template_class.
                    # Choose the last one so we can pop it efficiently.
                    else:
                        child_slot_name, child_slot_routes = (
                            parallel_slot_backlog_stack[-1][-1])
                        current_instance = (
                            instance_backlog_stack[-1][0].instance)
                        child_instances = getattr(
                            current_instance, child_slot_name)
                        child_index = itertools.count()

                        # The parallel path we chose has more steps on the way
                        # to the leaf node, so we need to continue deeper into
                        # the tree.
                        if child_slot_routes:
                            child_provenances = tuple(
                                TemplateProvenanceNode(
                                    parent_slot_key=child_slot_name,
                                    parent_slot_index=next(child_index),
                                    instance_id=id(child_instance),
                                    instance=child_instance)
                                for child_instance in child_instances)
                            instance_history_stack.append(child_provenances)
                            instance_backlog_stack.append(
                                deque(child_provenances))
                            parallel_slot_backlog_stack.append(
                                list(child_slot_routes))

                        # The parallel path we chose is actually a leaf node,
                        # which means that each child instance is a provenance.
                        else:
                            partial_provenance = tuple(
                                instance_backlog_level[0]
                                for instance_backlog_level
                                in instance_backlog_stack)

                            # Note that using extend here is basically just a
                            # shorthand for repeatedly iterating on the
                            # outermost while loop after appending the
                            # children (like we did with child_slot_routes)
                            provenances.extend(TemplateProvenance(
                                (
                                    *partial_provenance,
                                    TemplateProvenanceNode(
                                        parent_slot_key=child_slot_name,
                                        parent_slot_index=next(child_index),
                                        instance_id=id(child_instance),
                                        instance=child_instance)))
                                for child_instance
                                in child_instances)

                            # Note that we already popped from the parallel
                            instance_backlog_stack[-1].popleft()

                # Oh the humanity, oh the combinatorics!
                invocations.extend(itertools.product(
                    provenances,
                    itertools.chain.from_iterable(
                        abstract_calls.values()
                    )))

        root_provenance = TemplateProvenance((root_provenance_node,))
        root_template_class = type(root_template_instance)
        invocations.extend(
            (root_provenance, abstract_call)
            for abstract_call
            in itertools.chain.from_iterable(
                template_preload[root_template_class].function_calls.values()
            ))
        return invocations


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
    vars_ = {}
    content = {}
    for template_field in fields(cls):
        field_classification = _classify_interface_field_flavor(
            template_type_hints, template_field)

        # Note: it's not entirely clear to me that this restriction makes
        # sense; I could potentially see MAYBE there being some kind of
        # environment function that could access other attributes from the
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
            elif field_flavor is InterfaceAnnotationFlavor.SLOT:
                dest_lookup = slots
            else:
                dest_lookup = content

            dest_lookup[template_field.name] = wrapped_type

    cls._templatey_signature = TemplateSignature.new(
        slots=slots,
        vars_=vars_,
        content=content)
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
    """This is used by environment functions to indicate that a value is
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
            unescaped_vars_context: Mapping[str, object],
            parent_part_index: int,
            ) -> Iterable[str | InterpolatedVariable]:
        """Implement this for any instance of complex content. **Note
        that you should never perform the variable interpolation
        yourself.** Instead, you should do whatever content modification
        you need based on the variables, yielding back an
        InterpolatedVariable placeholder in place of the value. This
        lets templatey manage variable escaping, etc.
        """
        ...
