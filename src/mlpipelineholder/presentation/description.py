"""Pipeline rendering, output-conflict reporting, and result-history presentation."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.models import ExpressionRegistration, FunctionRegistration
from ..exceptions import ResolutionError

from ..core.base import PipelineBase
from ..execution.function_registry import inspect_exposed_input_names


def _is_child_pipeline(node: object) -> bool:
    return isinstance(node, PipelineBase)


class DescriptionMixin:
    """Presentation helpers: tree rendering, conflict reporting, and history."""

    if TYPE_CHECKING:
        registration_name: str = ""
        execution_priority: float | None = None
        gate_block: Any = None
        logger: Any = None
        historical_result_log_path: Any = None
        parent_pipeline: Any = None
        strict_mode: bool = False

        def _sorted_nodes(self) -> list[Any]: ...
        def _node_declared_outputs(self, node: Any) -> set[str]: ...
        def _priority_group(self, execution_priority: float | None) -> int: ...
        def _select_executable_node_in_group(self, nodes: list[Any]) -> Any: ...
        def qualified_node_name(self, node_name: str) -> str: ...
        def get_config_value(self, field_name: str) -> Any: ...
        def _declared_output_names_before_priority(self, priority: float | None) -> set[str]: ...
        def _visible_config_names(self) -> set[str]: ...

    def get_result_history(self) -> list[str]:
        # Attached child pipelines intentionally keep reading historical RESULT lines from
        # their own pre-attachment log path, while new runtime logging flows through the
        # parent logger. This preserves old child history but means nested logging is not
        # fully unified after attachment.
        if self.parent_pipeline is not None and self.historical_result_log_path is not None:
            attached_override = getattr(self, "_attached_result_history_override", None)
            if attached_override is not None:
                return list(attached_override)
            return self._read_result_history_from_file(self.historical_result_log_path)
        return self.logger.get_result_history()

    def print_result_history(self) -> None:
        for entry in self.get_result_history():
            print(self._color(entry, "green"))

    def clear_result_history(self) -> None:
        if self.parent_pipeline is not None and self.historical_result_log_path is not None:
            setattr(self, "_attached_result_history_override", [])
            return
        self.logger.clear_result_history()

    def get_priority_group(
        self, integer_priority: int
    ) -> tuple[list[str], str | None]:
        group_nodes = [
            node for node in self._sorted_nodes() if self._priority_group(node.execution_priority) == integer_priority
        ]
        names = [node.registration_name for node in group_nodes]
        executable = self._select_executable_node_in_group(group_nodes)
        return names, None if executable is None else executable.registration_name

    def get_output_conflicts(
        self,
    ) -> dict[str, dict[str, list[str] | str]]:
        conflicts: dict[str, dict[str, list[str] | str]] = {}
        seen: dict[str, str] = {}
        for node in self._sorted_nodes():
            producer_name = self.qualified_node_name(node.registration_name)
            for output_name in sorted(self._node_declared_outputs(node)):
                if output_name not in seen:
                    seen[output_name] = producer_name
                    continue
                conflict = conflicts.setdefault(
                    output_name,
                    {"created_by": seen[output_name], "overridden_by": []},
                )
                overridden_by = conflict["overridden_by"]
                if isinstance(overridden_by, list):
                    overridden_by.append(producer_name)
        return conflicts

    def describe_output_conflicts(self) -> str:
        lines = [f"Output conflicts in {self.registration_name}:"]
        conflicts = self.get_output_conflicts()
        if not conflicts:
            lines.append("- none")
        for output_name, data in sorted(conflicts.items()):
            lines.append(f"- {output_name}")
            lines.append(f"  first created by: {data['created_by']}")
            lines.append("  overridden by:")
            for producer in data["overridden_by"]:
                lines.append(f"    - {producer}")
        for node in self._sorted_nodes():
            if _is_child_pipeline(node):
                lines.append(node.describe_output_conflicts())
        return "\n".join(lines)

    def describe_pipeline(self) -> str:
        return "\n".join(self._describe_lines())

    def _describe_lines(
        self,
        indent: str = "",
        as_child: bool = False,
        muted: bool = False,
    ) -> list[str]:
        lines: list[str] = []
        symbol_color = "blue"
        # Arguments use pure black normally and dark grey when skipped.
        # termcolor's light_grey is SGR 37 (white), which is nearly invisible
        # on the white notebook background in Colab; dark_grey (SGR 90)
        # renders as a readable grey in both Jupyter and Colab.
        arg_color = "grey" if not muted else "dark_grey"
        muted = muted or (as_child and self._should_grey_in_chart())
        if as_child:
            line = f"{self._line_style(indent, muted)}{self._chart_color('pipeline', 'magenta', muted)} {self._chart_color(f'[{self.execution_priority}]', 'cyan', muted)} {self._chart_color(self.registration_name, 'blue', muted)}"
            lines.append(line)
        else:
            lines.append(
                f"{indent}{self._chart_color('PipelineHolder', 'green', muted)}{self._chart_color('(', symbol_color, muted)}{self._chart_color(self.registration_name, 'blue', muted)}{self._chart_color(')', symbol_color, muted)}"
            )
        if self.gate_block is not None:
            gate_args = self._chart_color(
                ", ".join(self._displayed_argument_names(self.gate_block.registration, None, None)),
                arg_color,
                muted,
            )
            gate_line = (
                f"{indent}{self._line_style('├── ', muted)}{self._chart_color('[gate]', 'magenta', muted)} {self._chart_color(self.gate_block.registration.function_name, 'green', muted)}"
                f"{self._chart_color('(', symbol_color, muted)}{gate_args}{self._chart_color(')', symbol_color, muted)}"
            )
            lines.append(gate_line)
        sorted_nodes = self._sorted_nodes()
        node_muted_states = [
            muted or (_is_child_pipeline(node) and node._should_grey_in_chart())
            for node in sorted_nodes
        ]
        for index, node in enumerate(sorted_nodes):
            is_last = index == len(sorted_nodes) - 1
            node_muted = node_muted_states[index]
            prefix = "└──" if is_last else "├──"
            # The spine segment owned by this level stays active while any
            # later sibling at this level will still run, so the outermost
            # vertical line never looks broken across a skipped subtree; it
            # only greys when this node and every later sibling is skipped.
            spine_muted = node_muted and all(node_muted_states[index + 1 :])
            child_indent = indent + self._spine_style(
                "    " if is_last else "│   ", spine_muted
            )
            if _is_child_pipeline(node):
                child_line = (
                    f"{indent}{self._line_style(f'{prefix} ', node_muted)}{self._chart_color('child-pipeline', 'magenta', node_muted)} {self._chart_color(f'[{node.execution_priority}]', 'cyan', node_muted)} {self._chart_color(node.registration_name, 'blue', node_muted)}"
                )
                lines.append(child_line)
                lines.extend(node._describe_lines(child_indent, as_child=True, muted=node_muted)[1:])
                continue
            block_line = (
                f"{indent}{self._line_style(f'{prefix} ', node_muted)}{self._chart_color(f'[{node.execution_priority}]', 'cyan', node_muted)} {self._chart_color(node.registration_name, 'blue', node_muted)}"
            )
            lines.append(block_line)
            for function_index, registration in enumerate(node.functions):
                function_prefix = "└──" if function_index == len(node.functions) - 1 else "├──"
                outputs = [
                    (
                        self._chart_color(f"{output_name}*", "red", node_muted)
                        if output_name in registration.save_to_disk
                        else self._chart_color(output_name, "green", node_muted)
                    )
                    for output_name in (
                        registration.output_names
                        if isinstance(registration, ExpressionRegistration)
                        else registration.produced_output_names
                    )
                ]
                args = self._chart_color(
                    ", ".join(
                        self._displayed_argument_names(
                            registration,
                            node.execution_priority,
                            node,
                        )
                    ),
                    arg_color,
                    node_muted,
                )
                function_line = (
                    f"{child_indent}{self._line_style(f'{function_prefix} ', node_muted)}{self._chart_color(registration.function_name, 'green', node_muted)}"
                    f"{self._chart_color('(', symbol_color, node_muted)}{args}{self._chart_color(')', symbol_color, node_muted)}"
                    + (
                        f" {self._chart_color('->', symbol_color, node_muted)} {', '.join(outputs)}"
                        if outputs
                        else ""
                    )
                )
                lines.append(function_line)
        return lines

    def _should_grey_in_chart(self) -> bool:
        if self.parent_pipeline is None or self.gate_block is None:
            return False
        config_field_name = self.gate_block.config_field_name
        if config_field_name is None:
            return False
        try:
            return self.get_config_value(config_field_name) != self.gate_block.expected_value
        except ResolutionError:
            return False

    def _chart_color(
        self, text: str, color: str, muted: bool
    ) -> str:
        swapped_normal_map = {
            "magenta": "light_magenta",
            "cyan": "light_cyan",
            "blue": "light_blue",
            "green": "light_green",
            "yellow": "light_yellow",
            "red": "light_red",
        }
        if not muted:
            return import_module("termcolor").colored(
                text,
                swapped_normal_map.get(color, color),
                force_color=True,
            )
        return import_module("termcolor").colored(
            text,
            color,
            force_color=True,
        )

    def _line_style(self, text: str, muted: bool) -> str:
        if not muted:
            return text
        return import_module("termcolor").colored(text, "light_grey", force_color=True)

    def _spine_style(self, text: str, muted: bool) -> str:
        """Color a spine segment (``│   `` or spaces) for the chart."""
        if not muted:
            return text
        return import_module("termcolor").colored(text, "light_grey", force_color=True)

    def _displayed_argument_names(
        self,
        registration: FunctionRegistration | ExpressionRegistration,
        priority: float | None,
        block: Any | None = None,
    ) -> list[str]:
        visible_output_names = self._declared_output_names_before_priority(priority)
        visible_config_names = self._visible_config_names()
        if block is not None and isinstance(registration, ExpressionRegistration):
            input_names = block._effective_expression_input_names(registration)
        elif block is not None and isinstance(registration, FunctionRegistration):
            # Display the callable's exposed signature (explicit parameters plus
            # block-scoped variadic helper names). ``registration.input_names``
            # tracks concrete variadic member dependencies for invalidation and
            # is intentionally not what the chart shows.
            input_names = inspect_exposed_input_names(
                registration.callable_obj,
                param_mapping=registration.param_mapping,
                var_pos_name=None
                if self.strict_mode
                else registration.var_pos_name,
                var_kw_name=None
                if self.strict_mode
                else registration.var_kw_name,
                strict_mode=self.strict_mode,
            )
        else:
            input_names = registration.input_names
        displayed = [
            name
            for name in input_names
            if name in visible_output_names or name in visible_config_names
        ]
        var_pos_name = getattr(registration, "var_pos_name", None)
        if (
            block is not None
            and var_pos_name is not None
            and var_pos_name in block.registered_args
        ):
            displayed.append(var_pos_name)
        var_kw_name = getattr(registration, "var_kw_name", None)
        if (
            block is not None
            and var_kw_name is not None
            and var_kw_name in block.registered_kwargs
        ):
            displayed.append(var_kw_name)
        return displayed

    def _read_result_history_from_file(
        self, file_path: str
    ) -> list[str]:
        path = Path(file_path)
        if not path.exists():
            return []
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if " RESULT " in line
        ]

    def _color(self, text: str, color: str) -> str:
        return import_module("termcolor").colored(text, color, force_color=True)
