"""Invalidation state: node-output erasure, cascade walks, and mirror sync."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..state.output_pointers import OutputAddress

_holder_base: type | None = None


def _register_invalidation_holder_base(holder_base: type) -> None:
    """Register the holder class so invalidation code can recognise child pipelines."""
    global _holder_base
    _holder_base = holder_base


def _is_child_pipeline(node: object) -> bool:
    return _holder_base is not None and isinstance(node, _holder_base)


class InvalidationMixin:
    """Erasing produced outputs and cascading invalidation through consumers."""

    if TYPE_CHECKING:
        producer_outputs: dict[str, dict[str, Any]] = {}
        registration_name: str = ""
        parent_pipeline: Any = None
        execution_priority: float | None = None
        logger: Any = None
        _invalidation_forbidden: bool = False

        def _public_output_slots(self) -> dict[OutputAddress, Any]: ...
        def _prepare_pointer_removal(
            self,
            invalidated: set[OutputAddress],
            slots: dict[OutputAddress, Any],
        ) -> None: ...
        def _root_pipeline(self) -> Any: ...
        def _refresh_pointer_visible_state(self) -> None: ...
        def _delete_artifacts_from_outputs(self, outputs: dict[str, Any]) -> None: ...
        def _rebuild_visible_state(
            self, upstream_outputs: dict[str, Any] | None = None
        ) -> None: ...
        def _incoming_parent_outputs(self) -> dict[str, Any]: ...
        def _sorted_nodes(self) -> list[Any]: ...
        def full_path(self) -> str: ...
        def _required_input_names(self, node: Any) -> set[str]: ...
        def _invalidate_from_priority(
            self,
            priority: float,
            include_target: bool = True,
            preserve_pipeline_state: Any = None,
        ) -> None: ...

    def _erase_node_outputs(self, node_name: str) -> None:
        slots = self._public_output_slots()
        removed_address_values = self.producer_outputs.get(node_name, {})
        invalidated = {
            OutputAddress(self.registration_name, node_name, output_name)
            for output_name in removed_address_values
        }
        for address, value in (
            (OutputAddress(self.registration_name, node_name, name), value)
            for name, value in removed_address_values.items()
        ):
            slots[address] = value
        self._prepare_pointer_removal(invalidated, slots)
        root = self._root_pipeline()
        for output_name, addresses in list(
            root._optuna_study_group_addresses.items()
        ):
            addresses.difference_update(invalidated)
            if not addresses:
                root._optuna_study_group_addresses.pop(output_name, None)
                root._optuna_study_group_names.pop(output_name, None)
        removed = self.producer_outputs.pop(node_name, {})
        self._refresh_pointer_visible_state()
        self._delete_artifacts_from_outputs(removed)

    def _erase_selected_node_outputs(
        self,
        node_name: str,
        output_names: list[str],
    ) -> None:
        outputs = self.producer_outputs.get(node_name)
        if outputs is None:
            return
        removed = {
            output_name: outputs.pop(output_name)
            for output_name in output_names
            if output_name in outputs
        }
        if not outputs:
            self.producer_outputs.pop(node_name, None)
        self._rebuild_visible_state(self._incoming_parent_outputs())
        self._delete_artifacts_from_outputs(removed)

    def _erase_overridden_node_outputs(
        self,
        node_name: str,
        old_priority: float | None,
        new_priority: float | None,
        old_output_names: list[str],
        new_output_names: list[str] | None = None,
        *,
        erase_output_names: list[str] | None = None,
    ) -> None:
        """Unified erasure for a forced override of an expression, function or atom.

        Always erases the overridden node's own produced outputs. Unless cascade
        invalidation is forbidden, it then erases from the earliest downstream
        block that consumes any affected output name: the old outputs (whose
        values change or disappear) and the new outputs (which may collide with
        downstream inputs). Downstream blocks consuming none of the affected
        names are left untouched. Old outputs are walked from the old priority
        and new outputs from the new priority, so a priority change still catches
        consumers between the two positions.
        """
        if erase_output_names is None:
            self._erase_node_outputs(node_name)
        else:
            self._erase_selected_node_outputs(node_name, erase_output_names)
        if self.parent_pipeline is not None:
            self._resync_mirror_to_parent()
        if self._invalidation_forbidden:
            return
        users: list[tuple[Any, Any]] = []
        affected_names: set[str] = set()
        for output_name in dict.fromkeys(old_output_names):
            name_users = self._downstream_input_users(old_priority, output_name)
            if name_users:
                users.extend(name_users)
                affected_names.add(output_name)
        for output_name in dict.fromkeys(new_output_names or []):
            name_users = self._downstream_input_users(new_priority, output_name)
            if name_users:
                users.extend(name_users)
                affected_names.add(output_name)
        if not users:
            return
        labels = sorted(
            {
                f"'{pipeline.full_path()}.{node.registration_name}'"
                for pipeline, node in users
            }
        )
        self.logger.warning(
            f"Output(s) '{', '.join(sorted(affected_names))}' of block '{node_name}' "
            f"are used as inputs by downstream block(s) {', '.join(labels)}; "
            "invalidating those blocks and everything downstream of them"
        )
        by_pipeline: dict[int, tuple[Any, float]] = {}
        for owning_pipeline, node in users:
            if node.execution_priority is None:
                continue
            current = by_pipeline.get(id(owning_pipeline))
            if current is None or node.execution_priority < current[1]:
                by_pipeline[id(owning_pipeline)] = (
                    owning_pipeline,
                    node.execution_priority,
                )
        for owning_pipeline, priority in by_pipeline.values():
            owning_pipeline._invalidate_with_ancestor_consumers(priority)
        resynced: set[int] = set()
        if self.parent_pipeline is not None:
            self._resync_mirror_to_parent()
            resynced.add(id(self))
        for owning_pipeline, _ in by_pipeline.values():
            if id(owning_pipeline) in resynced:
                continue
            if owning_pipeline.parent_pipeline is not None:
                owning_pipeline._resync_mirror_to_parent()
                resynced.add(id(owning_pipeline))

    def _downstream_input_users(
        self,
        block_priority: float | None,
        input_name: str,
    ) -> list[tuple[Any, Any]]:
        """Blocks anywhere downstream that consume ``input_name`` as an input.

        A consumer is only impacted when the expression is its effective source:
        the first downstream node that also produces ``input_name`` shields every
        consumer after it (later producers win in the visibility model), so the
        walk stops there. Covers blocks after the expression in its own pipeline,
        blocks in descendant pipelines, and blocks in ancestor pipelines after
        the child node on the path.
        """
        users: list[tuple[Any, Any]] = []
        self._walk_input_users_stopping_at_producer(
            self,
            block_priority,
            input_name,
            users,
        )
        if self._has_shielding_producer(block_priority, input_name):
            return users
        current: Any = self
        while current is not None and current.parent_pipeline is not None:
            parent = current.parent_pipeline
            child_node = next(
                (node for node in parent._sorted_nodes() if node is current),
                None,
            )
            if child_node is None or child_node.execution_priority is None:
                break
            self._walk_input_users_stopping_at_producer(
                parent,
                child_node.execution_priority,
                input_name,
                users,
            )
            if parent._has_shielding_producer(
                child_node.execution_priority,
                input_name,
            ):
                break
            current = parent
        return users

    def _invalidate_with_ancestor_consumers(self, priority: float) -> None:
        """Brutally invalidate from a consumer through every ancestor tail.

        The target pipeline loses the consumer and every later node. Each
        ancestor keeps the child node containing that consumer, but loses every
        node registered after that child, preserving one positional cutoff
        across nested pipeline boundaries.
        """
        self._invalidate_from_priority(priority)
        current: Any = self
        while current.parent_pipeline is not None:
            parent = current.parent_pipeline
            current._resync_mirror_to_parent()
            if current.execution_priority is None:
                return
            parent._invalidate_from_priority(
                current.execution_priority,
                include_target=False,
            )
            current = parent

    def _has_shielding_producer(
        self,
        block_priority: float | None,
        input_name: str,
    ) -> bool:
        """Whether a later node keeps ``input_name``'s effective value stable.

        A same-name producer after the overridden node wins in the visibility
        model, so removing or changing the node's output does not alter this
        pipeline's effective value — ancestor consumers of this pipeline's
        mirror are therefore not impacted. A producing node that itself consumes
        the name does not shield (its stored output was computed from the old
        value and will change).
        """
        for node in self._sorted_nodes():
            if (
                block_priority is None
                or node.execution_priority is None
                or node.execution_priority <= block_priority
            ):
                continue
            produced_outputs = self.producer_outputs.get(node.registration_name, {})
            if input_name in produced_outputs:
                if _is_child_pipeline(node):
                    return True
                return input_name not in self._block_consumed_input_names(node)
        return False

    def _walk_input_users_stopping_at_producer(
        self,
        pipeline: Any,
        start_priority: float | None,
        input_name: str,
        users: list[tuple[Any, Any]],
    ) -> None:
        """Walk one pipeline's nodes in priority order, stopping at the first producer.

        Nodes at or below ``start_priority`` are skipped (the expression itself and
        everything before it). The walk stops at the first node that produces
        ``input_name`` — that node's output overrides the expression for every
        later consumer, so later consumers are not impacted. A producing node that
        also consumes the name is still flagged (its stored output was computed
        from the old value). Nested pipeline nodes are recursed into: their blocks
        see the expression's output as upstream, so they can contain impacted
        consumers before their own first producer.
        """
        for node in pipeline._sorted_nodes():
            if (
                start_priority is not None
                and (
                    node.execution_priority is None
                    or node.execution_priority <= start_priority
                )
            ):
                continue
            produced_outputs = pipeline.producer_outputs.get(
                node.registration_name,
                {},
            )
            if input_name in produced_outputs:
                if (
                    not _is_child_pipeline(node)
                    and input_name in pipeline._block_consumed_input_names(node)
                ):
                    users.append((pipeline, node))
                if _is_child_pipeline(node):
                    self._walk_input_users_stopping_at_producer(
                        node,
                        None,
                        input_name,
                        users,
                    )
                break
            if _is_child_pipeline(node):
                self._walk_input_users_stopping_at_producer(
                    node,
                    None,
                    input_name,
                    users,
                )
                continue
            if input_name in pipeline._block_consumed_input_names(node):
                users.append((pipeline, node))

    def _block_consumed_input_names(self, block: Any) -> set[str]:
        names = set(self._required_input_names(block))
        for args_registration in block.registered_args.values():
            names.update(args_registration.ordered_items)
        for kwargs_registration in block.registered_kwargs.values():
            names.update(kwargs_registration.mapping_dct.values())
        return names

    def _resync_mirror_to_parent(self) -> None:
        current: Any = self
        while current is not None and current.parent_pipeline is not None:
            parent = current.parent_pipeline
            current_outputs = current._locally_produced_outputs()
            if current_outputs:
                parent.producer_outputs[current.registration_name] = current_outputs
            else:
                parent.producer_outputs.pop(current.registration_name, None)
            parent._rebuild_visible_state(parent._incoming_parent_outputs())
            current = parent

    def _locally_produced_outputs(self) -> dict[str, Any]:
        outputs: dict[str, Any] = {}
        for node in self._sorted_nodes():
            outputs.update(self.producer_outputs.get(node.registration_name, {}))
        return outputs
