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


if __name__ == "__main__":
    unittest.main()
