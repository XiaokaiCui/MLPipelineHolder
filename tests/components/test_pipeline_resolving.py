from __future__ import annotations

import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from mlpipelineholder import PipelineHandler, ResolutionError, pipeline_resolving


def produce_selected() -> str:
    return "pipeline-selected"


def produce_early() -> str:
    return "early"


def produce_late() -> str:
    return "late"


def produce_pair() -> tuple[str, str]:
    return "first", "second"


class PipelineResolvingTests(unittest.TestCase):
    def test_resolves_inputs_while_preserving_explicit_arguments_and_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = PipelineHandler(
                "investigation",
                {"splitting_date": "2026-09-23", "common_df": "pipeline-common"},
                Path(tmp) / "project",
            )
            pipeline.add_block("produce", 1).register_function(
                produce_selected,
                ["selected_df"],
            )
            pipeline.set_constant_value("stock_universe", ["AAPL", "MSFT"])
            pipeline.save_to_storage("stored_note", {"source": "storage"})
            pipeline.run_all()
            nodes_before = list(pipeline.nodes)
            explicit_common = object()

            @pipeline_resolving(
                pipeline,
                mapping={"note": "stored_note"},
            )
            def investigate(
                selected_df: str,
                common_df: object,
                splitting_date: str,
                stock_universe: list[str],
                note: dict[str, str],
                optional: str = "function-default",
            ) -> tuple[object, ...]:
                return (
                    selected_df,
                    common_df,
                    splitting_date,
                    stock_universe,
                    note,
                    optional,
                )

            with patch.object(
                pipeline,
                "_resolve_investigation_input",
                wraps=pipeline._resolve_investigation_input,
            ) as resolver:
                result = investigate(common_df=explicit_common)

            self.assertEqual(result[0], "pipeline-selected")
            self.assertIs(result[1], explicit_common)
            self.assertEqual(result[2], "2026-09-23")
            self.assertEqual(result[3], ["AAPL", "MSFT"])
            self.assertEqual(result[4], {"source": "storage"})
            self.assertEqual(result[5], "function-default")
            self.assertEqual(
                [call.args[0] for call in resolver.call_args_list],
                ["selected_df", "splitting_date", "stock_universe", "stored_note"],
            )
            self.assertEqual(pipeline.nodes, nodes_before)

    def test_inspect_resolves_multiple_values_for_one_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = PipelineHandler(
                "investigation",
                {"splitting_date": "2026-09-23"},
                Path(tmp) / "project",
            )
            pipeline.add_block("produce", 1).register_function(
                produce_selected,
                ["selected_df"],
            )
            pipeline.set_constant_value("stock_universe", ["AAPL", "MSFT"])
            pipeline.save_to_storage("stored_note", {"source": "storage"})
            pipeline.run_all()
            nodes_before = list(pipeline.nodes)

            inspection = pipeline.inspect(
                "selected_df",
                "splitting_date",
                "stock_universe",
                "stored_note",
            )
            with inspection as data:
                self.assertEqual(data.selected_df, "pipeline-selected")
                self.assertEqual(data["selected_df"], "pipeline-selected")
                self.assertEqual(data.splitting_date, "2026-09-23")
                self.assertEqual(data.stock_universe, ["AAPL", "MSFT"])
                self.assertEqual(data.stored_note, {"source": "storage"})
                self.assertIs(data, inspection)

            with self.assertRaisesRegex(RuntimeError, "only inside the with block"):
                _ = inspection["selected_df"]
            with self.assertRaisesRegex(RuntimeError, "cannot be reused"):
                inspection.__enter__()
            self.assertEqual(inspection._values, {})
            self.assertIsNone(inspection._pipeline)
            self.assertEqual(pipeline.nodes, nodes_before)

    def test_inspect_priority_uses_inputs_before_the_runtime_selected_node(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = PipelineHandler(
                "investigation",
                local_folder_path=Path(tmp) / "project",
            )
            pipeline.add_block("early", 1).register_function(
                produce_early,
                ["early_value"],
            )
            pipeline.add_block("selected", 5.1).register_function(
                produce_selected,
                ["selected_value"],
            )
            pipeline.add_block("late", 6).register_function(
                produce_late,
                ["late_value"],
            )
            pipeline.run_all()

            with patch.object(
                pipeline.logger,
                "info",
                wraps=pipeline.logger.info,
            ) as info:
                with pipeline.inspect("early_value", priority=5) as data:
                    self.assertEqual(data.early_value, "early")

            self.assertIn("selected", info.call_args.args[0])
            self.assertIn("requested priority 5", info.call_args.args[0])
            with self.assertRaises(ResolutionError):
                with pipeline.inspect("selected_value", priority=5):
                    self.fail("selected node output unexpectedly visible")
            with pipeline.inspect("selected_value", priority=6) as data:
                self.assertEqual(data.selected_value, "pipeline-selected")
            with pipeline.inspect("late_value") as data:
                self.assertEqual(data.late_value, "late")

    def test_priority_selection_uses_runtime_alternative_and_lower_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = PipelineHandler(
                "root",
                {"run_child": False},
                Path(tmp) / "root",
            )
            root.add_block("early", 1).register_function(
                produce_early,
                ["early_value"],
            )
            gated = PipelineHandler(
                "gated",
                local_folder_path=Path(tmp) / "gated",
            )
            gated.set_gate_block("run_child")
            gated.add_block("inside", 1).register_function(
                produce_selected,
                ["selected_value"],
            )
            root.add_child_pipeline(gated, 5.1)
            alternative = root.add_block("alternative", 5.2)
            alternative.register_function(produce_late, ["alternative_value"])
            root.run_all()

            self.assertEqual(
                root.get_selected_node_output("alternative_value", priority=5),
                "late",
            )

            fallback = PipelineHandler(
                "fallback",
                {"run_child": False},
                Path(tmp) / "fallback",
            )
            fallback.add_block("early", 1).register_function(
                produce_early,
                ["early_value"],
            )
            gated_only = PipelineHandler(
                "gated_only",
                local_folder_path=Path(tmp) / "gated_only",
            )
            gated_only.set_gate_block("run_child")
            gated_only.add_block("inside", 1).register_function(
                produce_selected,
                ["selected_value"],
            )
            fallback.add_child_pipeline(gated_only, 5.1)
            fallback.run_all()

            self.assertEqual(
                fallback.get_selected_node_output("early_value", priority=5),
                "early",
            )
            with self.assertRaisesRegex(ResolutionError, "No selectable node"):
                fallback.get_selected_node_output("early_value", priority=0)

    def test_get_selected_node_output_supports_name_priority_and_ordered_tuple(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = PipelineHandler(
                "root",
                local_folder_path=Path(tmp) / "root",
            )
            pair = root.add_block("pair", 1)
            pair.register_function(produce_pair, ["first_value", "second_value"])
            root.create_atom_child_pipeline(
                "atom",
                2,
                produce_selected,
                output_variable_names="atom_value",
            )
            child = PipelineHandler(
                "child",
                local_folder_path=Path(tmp) / "child",
            )
            child.add_block("inside", 1).register_function(
                produce_late,
                ["child_value"],
            )
            root.add_child_pipeline(child, 3)
            root.run_all()

            self.assertEqual(
                root.get_selected_node_output(
                    ["second_value", "first_value"],
                    priority=1,
                ),
                ("second", "first"),
            )
            self.assertEqual(
                root.get_selected_node_output(
                    "first_value",
                    priority=1,
                    node_name="pair",
                ),
                "first",
            )
            self.assertEqual(
                root.get_selected_node_output("atom_value", node_name="atom"),
                "pipeline-selected",
            )
            self.assertEqual(
                root.get_selected_node_output("child_value", node_name="child"),
                "late",
            )
            with self.assertRaisesRegex(ResolutionError, "selects node 'atom'"):
                root.get_selected_node_output(
                    "first_value",
                    priority=2,
                    node_name="pair",
                )

    def test_priority_apis_require_integer_priorities_and_valid_selectors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = PipelineHandler(
                "investigation",
                local_folder_path=Path(tmp) / "project",
            )
            pipeline.add_block("producer", 1).register_function(
                produce_selected,
                ["selected_value"],
            )
            pipeline.run_all()

            for invalid_priority in (True, 1.0, "1"):
                with self.assertRaisesRegex(TypeError, "must be an integer"):
                    pipeline.inspect("selected_value", priority=invalid_priority)  # type: ignore[arg-type]
                with self.assertRaisesRegex(TypeError, "must be an integer"):
                    pipeline.get_selected_node_output(
                        "selected_value",
                        priority=invalid_priority,  # type: ignore[arg-type]
                    )
            with self.assertRaisesRegex(ValueError, "priority or node_name"):
                pipeline.get_selected_node_output("selected_value")
            with self.assertRaisesRegex(TypeError, "string or list"):
                pipeline.get_selected_node_output(("selected_value",), node_name="producer")  # type: ignore[arg-type]

    def test_inspect_dictionary_access_handles_attribute_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = PipelineHandler(
                "investigation",
                local_folder_path=Path(tmp) / "project",
            )
            pipeline.save_to_storage("market-data", 7)
            pipeline.save_to_storage("__class__", "stored-class")

            with pipeline.inspect("market-data", "__class__") as data:
                self.assertEqual(data["market-data"], 7)
                self.assertEqual(data["__class__"], "stored-class")
                self.assertIsNot(data.__class__, str)
                with self.assertRaises(KeyError):
                    _ = data["missing"]
                with self.assertRaises(AttributeError):
                    _ = data.missing

    def test_inspect_validates_names_and_cleans_up_after_resolution_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = PipelineHandler(
                "investigation",
                local_folder_path=Path(tmp) / "project",
            )
            with self.assertRaisesRegex(ValueError, "at least one"):
                pipeline.inspect()
            with self.assertRaisesRegex(ValueError, "non-empty strings"):
                pipeline.inspect("")
            with self.assertRaisesRegex(ValueError, "must be unique"):
                pipeline.inspect("value", "value")

            inspection = pipeline.inspect("missing")
            with self.assertRaises(ResolutionError):
                inspection.__enter__()
            self.assertEqual(inspection._values, {})
            self.assertIsNone(inspection._pipeline)

    def test_inspect_cleans_up_after_body_failure_and_child_cannot_use_root_storage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = PipelineHandler(
                "root",
                {"config_value": 3},
                Path(tmp) / "root",
            )
            root.save_to_storage("root_only", {"source": "storage"})
            child = PipelineHandler(
                "child",
                local_folder_path=Path(tmp) / "child",
            )
            root.add_child_pipeline(child, 1)

            inspection = root.inspect("config_value")
            with self.assertRaisesRegex(RuntimeError, "body failed"):
                with inspection as data:
                    self.assertEqual(data.config_value, 3)
                    raise RuntimeError("body failed")
            self.assertEqual(inspection._values, {})
            self.assertIsNone(inspection._pipeline)

            with self.assertRaises(ResolutionError):
                with child.inspect("root_only"):
                    self.fail("child inspection unexpectedly resolved root storage")

    def test_atoms_and_blocks_expose_execution_inspect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = PipelineHandler(
                "root",
                local_folder_path=Path(tmp) / "root",
            )
            block = pipeline.add_block("regular", 1)
            pipeline.create_atom_child_pipeline(
                "atom",
                2,
                produce_selected,
                output_variable_names="selected_df",
            )
            atom = pipeline.get_child_pipeline("atom")

            self.assertTrue(hasattr(block, "inspect"))
            self.assertTrue(hasattr(atom, "inspect"))

    def test_storage_resolution_does_not_cache_an_unloaded_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = PipelineHandler(
                "investigation",
                local_folder_path=Path(tmp) / "project",
            )
            hash_id = pipeline.save_to_storage("stored_value", {"large": "value"})
            record = pipeline._stored_objects[hash_id]
            pipeline._persist_stored_object(record, pipeline.project_root)
            record.value = None
            record.value_is_loaded = False

            @pipeline_resolving(pipeline)
            def investigate(stored_value: dict[str, str]) -> dict[str, str]:
                return stored_value

            self.assertEqual(investigate(), {"large": "value"})
            self.assertFalse(record.value_is_loaded)
            self.assertIsNone(record.value)

            with pipeline.inspect("stored_value") as data:
                self.assertEqual(data.stored_value, {"large": "value"})
            self.assertFalse(record.value_is_loaded)
            self.assertIsNone(record.value)

    def test_missing_input_raises_resolution_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = PipelineHandler(
                "investigation",
                local_folder_path=Path(tmp) / "project",
            )

            @pipeline_resolving(pipeline)
            def investigate(missing_value: object) -> object:
                return missing_value

            with self.assertRaises(ResolutionError):
                investigate()

    @unittest.skipUnless(find_spec("dask.dataframe") is not None, "dask is not installed")
    def test_compute_only_materializes_auto_resolved_dask_values(self) -> None:
        import dask.dataframe as dd

        with tempfile.TemporaryDirectory() as tmp:
            pipeline = PipelineHandler(
                "investigation",
                local_folder_path=Path(tmp) / "project",
            )
            selected_df = dd.from_pandas(pd.DataFrame({"ticker": ["AAPL", "MSFT"]}), npartitions=1)
            explicit_common = dd.from_pandas(pd.DataFrame({"value": [1, 2]}), npartitions=1)
            pipeline.set_constant_value("selected_df", selected_df, copy=False)

            @pipeline_resolving(pipeline, compute=True)
            def investigate(selected_df: object, common_df: object) -> tuple[object, object]:
                return selected_df, common_df

            resolved_selected, received_common = investigate(common_df=explicit_common)

            self.assertIsInstance(resolved_selected, pd.DataFrame)
            self.assertIs(received_common, explicit_common)

            with pipeline.inspect("selected_df", compute=False) as data:
                self.assertIs(data.selected_df, selected_df)
            with pipeline.inspect("selected_df", compute=True) as data:
                self.assertIsInstance(data.selected_df, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
