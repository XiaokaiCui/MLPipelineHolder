from __future__ import annotations

import unittest
import warnings
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pandas as pd

from mlpipelineholder import PipelineHandler, ResolutionError, pipeline_resolving
from mlpipelineholder.persistence.artifacts.store import ArtifactStore


def return_value(value: object) -> object:
    return value


def produce_output() -> dict[str, str]:
    return {"source": "output"}


def append_value(value: list[int]) -> list[int]:
    value.append(99)
    return value


class InspectionCacheTests(unittest.TestCase):
    def _pipeline(self, root: Path) -> PipelineHandler:
        pipeline = PipelineHandler("inspection-cache", {}, root)
        pipeline.set_constant_value("first", [1, 2], to_disk=True)
        pipeline.set_constant_value("second", {"value": 3}, to_disk=True)
        return pipeline

    def test_incremental_loading_reuses_shared_values_and_selectively_refreshes(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = self._pipeline(Path(tmp))
            with mock.patch.object(
                ArtifactStore,
                "load",
                wraps=pipeline.artifact_store.load,
            ) as load:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    pipeline.load_for_inspection(["first"])
                self.assertEqual(load.call_count, 1)

                cached: list[int] = []
                with pipeline.inspect("first") as resolved:
                    cached = resolved.first
                    cached.append(4)
                with pipeline.inspect("first") as resolved:
                    self.assertIs(resolved.first, cached)
                    self.assertEqual(resolved.first, [1, 2, 4])

                @pipeline_resolving(pipeline)
                def investigate(first: list[int]) -> list[int]:
                    return first

                self.assertIs(investigate(), cached)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    pipeline.load_for_inspection(["second"])
                self.assertEqual(load.call_count, 2)
                with pipeline.inspect("first") as resolved:
                    self.assertIs(resolved.first, cached)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    pipeline.load_for_inspection(["first"])
                self.assertEqual(load.call_count, 3)
                with pipeline.inspect("first") as resolved:
                    self.assertIsNot(resolved.first, cached)
                    self.assertEqual(resolved.first, [1, 2])

    def test_refresh_and_unload_apply_to_all_loaded_objects(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = self._pipeline(Path(tmp))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pipeline.load_for_inspection(["first", "second"])

            with pipeline.inspect("first", "second") as resolved:
                resolved.first.append(8)
                resolved.second["value"] = 9
            pipeline.refresh_loaded_objects()
            with pipeline.inspect("first", "second") as resolved:
                self.assertEqual(resolved.first, [1, 2])
                self.assertEqual(resolved.second, {"value": 3})

            pipeline.unload_inspection_objects()
            self.assertEqual(pipeline._inspection_cache_bindings, {})
            with pipeline.inspect("first") as resolved:
                self.assertEqual(resolved.first, [1, 2])
            with self.assertWarnsRegex(UserWarning, "nothing was unloaded"):
                pipeline.unload_inspection_objects()
            with self.assertWarnsRegex(UserWarning, "nothing was refreshed"):
                pipeline.refresh_loaded_objects()

    def test_invalid_batch_does_not_partially_populate_cache(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = self._pipeline(Path(tmp))
            pipeline.set_constant_value("memory_only", [4, 5])

            with self.assertRaisesRegex(ResolutionError, "requires disk-backed"):
                pipeline.load_for_inspection(["first", "memory_only"])

            self.assertEqual(pipeline._inspection_cache_bindings, {})
            with self.assertRaisesRegex(ValueError, "must be unique"):
                pipeline.load_for_inspection(["first", "first"])
            with self.assertRaisesRegex(ValueError, "non-empty"):
                pipeline.load_for_inspection([])
            with self.assertRaisesRegex(TypeError, "list or tuple"):
                pipeline.load_for_inspection(
                    "first"  # pyright: ignore[reportArgumentType]
                )

    def test_normal_getters_and_execution_bypass_inspection_cache(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = self._pipeline(Path(tmp))
            block = pipeline.add_block("consume", 1)
            block.register_function(
                append_value,
                ["result"],
                param_mapping={"value": "first"},
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pipeline.load_for_inspection(["first"])
            with pipeline.inspect("first") as resolved:
                resolved.first.append(7)

            self.assertEqual(pipeline.get_constant_value("first"), [1, 2])
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("result"), [1, 2, 99])
            with pipeline.inspect("first") as resolved:
                self.assertEqual(resolved.first, [1, 2, 7])

    def test_inspect_node_bypasses_inspection_cache(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = self._pipeline(Path(tmp))
            block = pipeline.add_block("consume", 1)
            block.register_function(
                return_value,
                ["result"],
                param_mapping={"value": "first"},
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pipeline.load_for_inspection(["first"])
            with pipeline.inspect("first") as resolved:
                resolved.first.append(7)

            with mock.patch.object(
                ArtifactStore,
                "load",
                wraps=pipeline.artifact_store.load,
            ) as load:
                inspection = pipeline.inspect_node(
                    node_name="consume",
                    resolve_only=True,
                )
                self.assertEqual(load.call_count, 1)

            self.assertEqual(inspection.arguments["value"], [1, 2])

    def test_new_artifact_generation_bypasses_stale_cache(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = self._pipeline(Path(tmp))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pipeline.load_for_inspection(["first"])
            stale: list[int] = []
            with pipeline.inspect("first") as resolved:
                stale = resolved.first

            pipeline.set_constant_value("first", [10], to_disk=True)
            with pipeline.inspect("first") as resolved:
                self.assertIsNot(resolved.first, stale)
                self.assertEqual(resolved.first, [10])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pipeline.load_for_inspection(["first"])
            with pipeline.inspect("first") as resolved:
                self.assertEqual(resolved.first, [10])

    def test_preloads_disk_backed_outputs_and_root_storage(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline = PipelineHandler("sources", {}, root / "project")
            block = pipeline.add_block("produce", 1)
            block.register_function(
                produce_output,
                ["output_value"],
                save_to_disk=["output_value"],
            )
            pipeline.run_all()
            hash_id = pipeline.save_to_storage("stored", {"source": "storage"})
            pipeline.update_storage(
                hash_id,
                {"source": "storage"},
                to_disk=True,
            )

            with (
                mock.patch.object(
                    ArtifactStore,
                    "load",
                    wraps=pipeline.artifact_store.load,
                ) as load,
                warnings.catch_warnings(),
            ):
                warnings.simplefilter("ignore", UserWarning)
                pipeline.load_for_inspection(["output_value", "stored"])
                self.assertEqual(load.call_count, 2)

                with pipeline.inspect("output_value", "stored") as resolved:
                    self.assertEqual(resolved.output_value, {"source": "output"})
                    self.assertEqual(resolved.stored, {"source": "storage"})
                self.assertEqual(load.call_count, 2)

            pipeline.unload_inspection_objects()
            with self.assertWarnsRegex(UserWarning, "nothing was unloaded"):
                pipeline.unload_inspection_objects()

    def test_preloads_config_artifact_records(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline = PipelineHandler("config-source", {}, root / "project")
            config_record = pipeline.artifact_store.save(
                variable_name="frame",
                value={"source": "config"},
                block_name="setup",
                function_name="set_config",
                run_id="run",
            )
            pipeline.set_config("frame", config_record)

            with (
                mock.patch.object(
                    ArtifactStore,
                    "load",
                    wraps=pipeline.artifact_store.load,
                ) as load,
                warnings.catch_warnings(),
            ):
                warnings.simplefilter("ignore", UserWarning)
                pipeline.load_for_inspection(["frame"])
                self.assertEqual(load.call_count, 1)

                with pipeline.inspect("frame") as resolved:
                    self.assertEqual(resolved.frame, {"source": "config"})
                self.assertEqual(load.call_count, 1)

    def test_cache_is_transient_across_save_and_load(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline = self._pipeline(root / "project")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pipeline.load_for_inspection(["first"])
            bundle = root / "bundle"
            pipeline.save_pipeline(bundle)

            loaded = PipelineHandler.load_pipeline(
                bundle,
                forced_deleting=True,
                trust_project=True,
            )

            self.assertEqual(loaded._inspection_cache_bindings, {})
            self.assertEqual(loaded._inspection_cache_entries, {})

    def test_memory_delta_logging_uses_process_rss_difference(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = self._pipeline(Path(tmp))
            with (
                mock.patch(
                    "mlpipelineholder.execution.inspection._process_rss_bytes",
                    side_effect=[100, 150],
                ),
                mock.patch.object(pipeline.logger, "info") as info,
                warnings.catch_warnings(),
            ):
                warnings.simplefilter("ignore", UserWarning)
                pipeline.load_for_inspection(["first"])
            info.assert_called_with("Inspection cache load RSS delta: +50 B")

            with (
                mock.patch(
                    "mlpipelineholder.execution.inspection._process_rss_bytes",
                    side_effect=[150, 120],
                ),
                mock.patch.object(pipeline.logger, "info") as info,
            ):
                pipeline.unload_inspection_objects()
            info.assert_called_with("Inspection cache unload RSS released: +30 B")

    @unittest.skipUnless(
        find_spec("dask.dataframe") is not None,
        "dask is not installed",
    )
    def test_dask_compute_configuration_can_be_replaced(self) -> None:
        import dask.dataframe as dd

        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("dask-cache", {}, Path(tmp))
            dask_frame = dd.from_pandas(
                pd.DataFrame({"value": [1, 2, 3]}),
                npartitions=1,
            )
            pipeline.set_constant_value("frame", dask_frame, to_disk=True)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pipeline.load_for_inspection(["frame"])
            with pipeline.inspect("frame", compute=False) as resolved:
                self.assertIsInstance(resolved.frame, pd.DataFrame)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pipeline.load_for_inspection(["frame"], compute=False)
            with pipeline.inspect("frame", compute=False) as resolved:
                self.assertIsInstance(resolved.frame, dd.DataFrame)
            with pipeline.inspect("frame", compute=True) as resolved:
                self.assertIsInstance(resolved.frame, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
