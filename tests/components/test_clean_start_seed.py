from __future__ import annotations

import pickle
import random
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
import unittest
from unittest.mock import patch

import numpy as np

from mlpipelineholder import PipelineHolder, RegistrationError
from mlpipelineholder.core.models import ArtifactRecord, RuntimeValueReference


def draw(scale: int, items: list[int]) -> tuple[float, int, int]:
    items.append(10)
    return random.random(), int(np.random.randint(100_000)), scale + len(items)


def read_items(items: list[int]) -> int:
    return len(items)


def always_true() -> bool:
    return True


def optional_draw() -> float:
    return random.random()


class CleanStartSeedTests(unittest.TestCase):
    def test_seeded_clean_run_restores_config_and_constant_values(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHolder("root", {"scale": 2}, Path(tmp) / "root")
            pipeline.set_constant_value("items", [1])
            pipeline.add_block("drawing", 1).register_function(draw, ["py", "np", "score"])
            pipeline.set_seed(123)
            pipeline.run_all()
            expected = tuple(pipeline.get_value(name) for name in ("py", "np", "score"))
            self.assertEqual(expected[-1], 4)
            pipeline.set_config("scale", 10)
            pipeline.set_constant_value("items", [9, 9])
            pipeline.set_constant_value("late", 3)
            pipeline.run_all(clean_start=True)
            self.assertEqual(tuple(pipeline.get_value(name) for name in ("py", "np", "score")), expected)
            self.assertEqual(pipeline.get_config_value("scale"), 2)
            self.assertEqual(pipeline.manual_values["items"], [1, 10])
            self.assertNotIn("late", pipeline.manual_values)
            pipeline.run_all(clean_start=True)
            self.assertEqual(tuple(pipeline.get_value(name) for name in ("py", "np", "score")), expected)
            self.assertEqual(pipeline.manual_values["items"], [1, 10])
            self.assertGreaterEqual(len(pipeline.run_history), 3)

    def test_first_child_execution_snapshots_sibling_and_child_clean_scope(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {"parent_config": 1}, Path(tmp) / "root")
            left = PipelineHolder("left", {"own": 1}, Path(tmp) / "left")
            right = PipelineHolder("right", {"own": 2}, Path(tmp) / "right")
            left.set_constant_value("items", [1])
            right.set_constant_value("items", [1, 2])
            left.add_block("count", 1).register_function(read_items, ["left_count"])
            right.add_block("count", 1).register_function(read_items, ["right_count"])
            root.add_child_pipeline(left, 1)
            root.add_child_pipeline(right, 2)
            root.set_seed(5)
            root.run_block("left")
            right.set_constant_value("items", [9])
            right.set_constant_value("late", 3)
            root.set_constant_value("root_late", 5)
            left.set_config("own", 7)
            root.set_config("parent_config", 8)
            left.run_all(clean_start=True)
            self.assertEqual(left.get_config_value("own"), 1)
            self.assertEqual(root.get_config_value("parent_config"), 8)
            self.assertEqual(right.manual_values["items"], [9])
            root.run_all(clean_start=True)
            self.assertEqual(root.get_config_value("parent_config"), 1)
            self.assertEqual(right.manual_values["items"], [1, 2])
            self.assertNotIn("late", right.manual_values)
            self.assertNotIn("root_late", root.manual_values)

    def test_manual_root_snapshot_replaces_all_constant_baselines(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {}, Path(tmp) / "root")
            child = PipelineHolder("child", {}, Path(tmp) / "child")
            root.add_child_pipeline(child, 1)
            root.set_constant_value("item", [1])
            child.set_constant_value("child_item", [2])
            root.snapshot_constants()
            root.set_constant_value("item", [3])
            child.set_constant_value("child_item", [4])
            root.snapshot_constants()
            root.set_constant_value("item", [5])
            child.set_constant_value("later", [6])
            root.run_all(clean_start=True)
            self.assertEqual(root.manual_values["item"], [3])
            self.assertEqual(child.manual_values["child_item"], [4])
            self.assertNotIn("later", child.manual_values)
            with self.assertRaises(RegistrationError):
                child.snapshot_constants()

    def test_unrestorable_constant_does_not_block_ordinary_runs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {}, Path(tmp) / "root")
            root.set_constant_value("bad", Lock(), copy=False)
            root.add_block("draw", 1).register_function(optional_draw, ["result"])
            root.run_all()
            self.assertIsNotNone(root.get_value("result"))
            self.assertIsNotNone(root._clean_baseline_error)
            with self.assertRaisesRegex(RegistrationError, "Clean baseline unavailable"):
                root.run_all(clean_start=True)
            root.set_constant_value("bad", [1])
            root.snapshot_constants()
            root.run_all(clean_start=True)
            self.assertEqual(root.get_constant_value("bad"), [1])

    def test_clean_start_rejects_invalidation_forbidden_before_mutation(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {}, Path(tmp) / "root")
            root.add_block("draw", 1).register_function(optional_draw, ["value"])
            root.run_all()
            before = root.get_value("value")
            root.forbid_invalidate_objects()
            with self.assertRaisesRegex(RegistrationError, "invalidation is forbidden"):
                root.run_all(clean_start=True)
            self.assertEqual(root.get_value("value"), before)

    def test_clean_start_bypasses_precleanup_gate_and_seeded_parallel_warns(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {"gate": True}, Path(tmp) / "root")
            root.add_block("first", 1).register_function(optional_draw, ["left"])
            root.add_block("second", 2).register_function(always_true, ["right"])
            root.set_gate_block("gate", True)
            root.set_seed(4)
            root.run_all()
            root.set_config("gate", False)
            root.gate_cleanup_confirmation = True
            with patch("builtins.input", side_effect=AssertionError("prompted")):
                root.run_all(clean_start=True)
            self.assertEqual(root.get_config_value("gate"), True)

    def test_save_load_preserves_seed_and_snapshots(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {"scale": 2}, Path(tmp) / "root")
            root.set_constant_value("items", [1])
            root.add_block("drawing", 1).register_function(draw, ["py", "np", "score"])
            root.set_seed(42)
            root.run_all()
            expected = root.get_value("np")
            root.set_config("scale", 9)
            root.set_constant_value("items", [8])
            root.save_pipeline(Path(tmp) / "bundle")
            loaded = PipelineHolder.load_pipeline(Path(tmp) / "bundle", forced_deleting=True, trust_project=True)
            self.assertEqual(loaded._seed, 42)
            loaded.run_all(clean_start=True)
            self.assertEqual(loaded.get_value("np"), expected)
            self.assertEqual(loaded.get_config_value("scale"), 2)
            self.assertEqual(loaded.manual_values["items"], [1, 10])


def reseed_user() -> float:
    random.seed(999)
    np.random.seed(999)
    return random.random()


class CleanStartAdditionalTests(unittest.TestCase):
    def test_custom_reseed_controls_downstream_without_per_node_reset(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {}, Path(tmp) / "root")
            root.add_block("user", 1).register_function(reseed_user, ["user_draw"])
            root.add_block("later", 2).register_function(optional_draw, ["later_draw"])
            root.set_seed(20)
            root.run_all(clean_start=True)
            first = root.get_value("later_draw")
            root.run_all(clean_start=True)
            self.assertEqual(root.get_value("later_draw"), first)
            independent = random.Random(999)
            independent.random()
            self.assertEqual(first, independent.random())

    def test_manual_snapshot_failure_keeps_previous_baseline(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {}, Path(tmp) / "root")
            root.set_constant_value("kept", [1])
            root.snapshot_constants()
            before = dict(root._clean_constant_baseline or {})
            root.manual_values["legacy_bad"] = Lock()
            with self.assertRaisesRegex(RegistrationError, "cannot be independently snapshotted"):
                root.snapshot_constants()
            self.assertEqual(root._clean_constant_baseline, before)
            root.run_all(clean_start=True)
            self.assertEqual(root.manual_values, {"kept": [1]})

    def test_parallel_warning_when_seed_or_second_function_added(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {}, Path(tmp) / "root")
            first = root.add_block("parallel", 1)
            first.register_function(optional_draw, ["left"])
            root.set_seed(12)
            with patch.object(root.logger, "warning") as log:
                first.register_function(always_true, ["right"])
            self.assertTrue(any("parallel" in str(call) and "not be reproducible" in str(call)
                                for call in log.call_args_list))
            with patch.object(root.logger, "warning") as log:
                root.set_seed(12)
            self.assertTrue(any("parallel" in str(call) for call in log.call_args_list))
            root.set_seed(None)
            with patch.object(root.logger, "warning") as log:
                root.run_all(clean_start=False)
            self.assertFalse(any("reproducible" in str(call) for call in log.call_args_list))

    def test_clean_gate_skip_does_not_prompt_with_disk_constant(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {"gate": False}, Path(tmp) / "root")
            root.set_constant_value("items", [1], to_disk=True)
            root.add_block("draw", 1).register_function(optional_draw, ["value"])
            root.set_gate_block("gate", True)
            root.set_config("gate", True)
            root.run_all()
            root.gate_cleanup_confirmation = True
            with patch("builtins.input", side_effect=AssertionError("prompted")):
                record = root.run_all(clean_start=True)
            self.assertEqual(record.status, "skipped")
            self.assertEqual(root.get_constant_value("items"), [1])
            self.assertIsNone(root._gate_cleanup_predecided)


def torch_draw() -> float:
    import torch
    return float(torch.rand(()))


class CleanStartOptionalLibraryTests(unittest.TestCase):
    def test_pandas_sampling_uses_seeded_numpy_global_rng(self) -> None:
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")
        def sample() -> list[int]:
            return list(pd.DataFrame({"item": list(range(100))}).sample(n=8).index)
        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {}, Path(tmp) / "root")
            root.add_block("sample", 1).register_function(sample, ["sampled"])
            root.set_seed(12)
            root.run_all(clean_start=True)
            first = root.get_value("sampled")
            root.run_all(clean_start=True)
            self.assertEqual(root.get_value("sampled"), first)

    def test_torch_global_seed_and_optional_deterministic_flags(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed")
        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {}, Path(tmp) / "root")
            root.add_block("torch", 1).register_function(torch_draw, ["draw"])
            prior_mode = torch.are_deterministic_algorithms_enabled()
            prior_benchmark = torch.backends.cudnn.benchmark
            root.set_seed(25, deterministic_torch=True)
            root.run_all(clean_start=True)
            first = root.get_value("draw")
            self.assertEqual(torch.are_deterministic_algorithms_enabled(), prior_mode)
            self.assertEqual(torch.backends.cudnn.benchmark, prior_benchmark)
            root.run_all(clean_start=True)
            self.assertEqual(root.get_value("draw"), first)


class CleanStartLifecycleTests(unittest.TestCase):
    def test_late_attached_child_gets_its_own_initial_snapshot(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {}, Path(tmp) / "root")
            root.add_block("first", 1).register_function(optional_draw, ["first_output"])
            root.run_all()
            child = PipelineHolder("child", {}, Path(tmp) / "child")
            child.set_constant_value("items", [3])
            child.add_block("count", 1).register_function(read_items, ["count"])
            root.add_child_pipeline(child, 2)
            root.run_all()
            child.set_constant_value("items", [4, 5])
            root.run_all(clean_start=True)
            self.assertEqual(child.manual_values["items"], [3])
            self.assertEqual(child.get_value("count"), 1)

    def test_each_partial_run_can_trigger_tree_baseline(self) -> None:
        for method in ("run_block", "run_from", "run_until"):
            with self.subTest(method=method), TemporaryDirectory() as tmp:
                root = PipelineHolder("root", {}, Path(tmp) / "root")
                root.set_constant_value("items", [1])
                root.add_block("count", 1).register_function(read_items, ["count"])
                getattr(root, method)("count")
                root.set_constant_value("items", [4, 5])
                root.run_all(clean_start=True)
                self.assertEqual(root.manual_values["items"], [1])
                self.assertEqual(root.get_value("count"), 1)

    def test_seed_is_root_only_and_disabled_by_default(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {}, Path(tmp) / "root")
            child = PipelineHolder("child", {}, Path(tmp) / "child")
            root.add_child_pipeline(child, 1)
            self.assertIsNone(root._seed)
            with self.assertRaises(RegistrationError):
                child.set_seed(1)
            with self.assertRaises(ValueError):
                root.set_seed(-1)
            with self.assertRaises(ValueError):
                root.set_seed(True)
            root.set_seed(0)
            root.set_seed(None)
            self.assertIsNone(root._seed)

    def test_inspection_cache_is_cleared_but_storage_and_history_remain(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {}, Path(tmp) / "root")
            root.set_constant_value("items", [1])
            hash_id = root.save_to_storage("outside", {"stored": True})
            root.add_block("count", 1).register_function(read_items, ["count"])
            root.run_all()
            root.copy_for_inspection(["items"])
            self.assertTrue(root._inspection_cache_bindings)
            root.run_all(clean_start=True)
            self.assertFalse(root._inspection_cache_bindings)
            self.assertTrue(root._stored_objects.get(hash_id))
            self.assertEqual(len(root.run_history), 2)


class CleanStartArtifactTests(unittest.TestCase):
    def test_disk_backed_constants_restore_values_and_retire_replaced_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {}, Path(tmp) / "root")
            root.set_constant_value("items", [1], to_disk=True)
            root.add_block("count", 1).register_function(read_items, ["count"])
            root.run_all()
            root.set_constant_value("items", [8, 9], to_disk=True)
            root.set_constant_value("late", [3], to_disk=True)
            old_items = Path(root.manual_values["items"].file_path)
            old_late = Path(root.manual_values["late"].file_path)
            root.run_all(clean_start=True)
            self.assertEqual(root.get_constant_value("items"), [1])
            self.assertIsInstance(root.manual_values["items"], ArtifactRecord)
            self.assertEqual(root.get_value("count"), 1)
            self.assertNotIn("late", root.manual_values)
            self.assertFalse(old_items.exists())
            self.assertFalse(old_late.exists())


class CleanStartStudyTests(unittest.TestCase):
    def test_optuna_constant_uses_detached_trial_snapshot(self) -> None:
        try:
            import optuna
        except ImportError:
            self.skipTest("Optuna not installed")
        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {}, Path(tmp) / "root")
            study = optuna.create_study(study_name="baseline", sampler=optuna.samplers.RandomSampler(seed=7))
            study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=1)
            root.set_constant_value("study", study)
            root.run_all()
            current = root.get_constant_value("study")
            current.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=1)
            self.assertEqual(len(current.trials), 2)
            root.run_all(clean_start=True)
            self.assertEqual(len(root.get_constant_value("study").trials), 1)
            root.run_all(clean_start=True)
            self.assertEqual(len(root.get_constant_value("study").trials), 1)


class CleanStartPreflightTests(unittest.TestCase):
    def test_restored_ancestor_config_collision_is_caught_before_mutation(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {"shared": 1}, Path(tmp) / "root")
            root.config.pop("shared")
            child = PipelineHolder("child", {}, Path(tmp) / "child")
            child.set_constant_value("shared", 3)
            root.add_child_pipeline(child, 1)
            with self.assertRaisesRegex(RegistrationError, "conflicts with current registration"):
                root.snapshot_constants()
            self.assertIsNone(root._clean_constant_baseline)
            with self.assertRaisesRegex(RegistrationError, "conflicts with current registration"):
                root.run_all(clean_start=True)
            self.assertEqual(root.config, {})
            self.assertEqual(child.get_constant_value("shared"), 3)


def draw_from_generator(generator: np.random.Generator) -> int:
    return int(generator.integers(0, 100_000))


class CleanStartGeneratorTests(unittest.TestCase):
    def test_pipeline_owned_generator_is_restored_from_constant_baseline(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {}, Path(tmp) / "root")
            external_generator = np.random.default_rng(1234)
            root.set_constant_value("generator", external_generator, copy=False)
            root.add_block("draw", 1).register_function(draw_from_generator, ["result"])
            root.set_seed(42)
            root.run_all()
            expected = root.get_value("result")
            root.run_all(clean_start=True)
            self.assertEqual(root.get_value("result"), expected)
            self.assertIsNot(root.manual_values["generator"], external_generator)
            root.run_all(clean_start=True)
            self.assertEqual(root.get_value("result"), expected)


class SeededAttachmentTests(unittest.TestCase):
    def test_attaching_parallel_child_to_seeded_root_warns(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {}, Path(tmp) / "root")
            root.set_seed(42)
            child = PipelineHolder("child", {}, Path(tmp) / "child")
            block = child.add_block("parallel", 1)
            block.register_function(optional_draw, ["first"])
            block.register_function(always_true, ["second"])
            with patch.object(root.logger, "warning") as warning:
                root.add_child_pipeline(child, 1)
            self.assertTrue(any("may not be reproducible" in str(call) for call in warning.call_args_list))


class CleanStartDaskTests(unittest.TestCase):
    def test_managed_dask_constant_survives_original_parquet_retirement(self) -> None:
        try:
            import dask.dataframe as dd
            import pandas as pd
        except ImportError:
            self.skipTest("Dask DataFrame and pandas not installed")
        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {}, Path(tmp) / "root")
            source = dd.from_pandas(pd.DataFrame({"score": [1, 2, 3]}), npartitions=2)
            root.set_constant_value("frame", source, to_disk=True)
            root.run_all()
            old_file = Path(root.manual_values["frame"].file_path)
            updated = dd.from_pandas(pd.DataFrame({"score": [9]}), npartitions=1)
            root.set_constant_value("frame", updated, to_disk=True)
            self.assertFalse(old_file.exists())
            root.run_all(clean_start=True)
            restored = root.get_constant_value("frame")
            self.assertIsInstance(restored, dd.DataFrame)
            self.assertEqual(restored.compute()["score"].tolist(), [1, 2, 3])


class CleanStartStudyPersistenceTests(unittest.TestCase):
    def test_saved_study_baseline_restores_original_trials(self) -> None:
        try:
            import optuna
        except ImportError:
            self.skipTest("Optuna not installed")
        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {}, Path(tmp) / "root")
            study = optuna.create_study(study_name="persisted_baseline")
            study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=1)
            root.set_constant_value("study", study)
            root.run_all()
            root.save_pipeline(Path(tmp) / "bundle")
            loaded = PipelineHolder.load_pipeline(Path(tmp) / "bundle", forced_deleting=True, trust_project=True)
            loaded.get_constant_value("study").optimize(
                lambda trial: trial.suggest_float("x", 0, 1), n_trials=1,
            )
            loaded.run_all(clean_start=True)
            self.assertEqual(len(loaded.get_constant_value("study").trials), 1)


class CleanStartInvalidCallTests(unittest.TestCase):
    def test_invalid_target_does_not_freeze_the_automatic_baseline(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {}, Path(tmp) / "root")
            for method in ("run_block", "run_from", "run_until"):
                with self.subTest(method=method), self.assertRaises(RegistrationError):
                    getattr(root, method)("missing")
                self.assertIsNone(root._clean_constant_baseline)
            root.set_constant_value("items", [1])
            root.add_block("count", 1).register_function(read_items, ["count"])
            root.run_all()
            self.assertIn("items", root._clean_constant_baseline or {})


class CleanStartTorchModuleTests(unittest.TestCase):
    def test_pipeline_owned_torch_module_restores_original_parameters(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed")
        def train(model: torch.nn.Linear) -> float:
            with torch.no_grad():
                model.weight.add_(1)
            return float(model.weight.item())
        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {}, Path(tmp) / "root")
            model = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                model.weight.fill_(1)
            root.set_constant_value("model", model, copy=False)
            root.add_block("train", 1).register_function(train, ["value"])
            root.set_seed(42)
            root.run_all()
            self.assertEqual(root.get_value("value"), 2.0)
            root.run_all(clean_start=True)
            self.assertEqual(root.get_value("value"), 2.0)
            self.assertIsNot(root.get_constant_value("model"), model)
            root.run_all(clean_start=True)
            self.assertEqual(root.get_value("value"), 2.0)


class CleanStartConstantNameTests(unittest.TestCase):
    def test_snapshot_rejects_name_conflicting_with_original_config(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {"value": 1}, Path(tmp) / "root")
            root.config.pop("value")
            root.set_constant_value("value", 2)
            with self.assertRaisesRegex(RegistrationError, "conflicts with current registration"):
                root.snapshot_constants()
            self.assertEqual(root.get_constant_value("value"), 2)


class CleanStartLegacyBaselineTests(unittest.TestCase):
    def test_legacy_placeholder_requires_replacement_before_clean_start(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {}, Path(tmp) / "root")
            placeholder = RuntimeValueReference("Lock", "<lock>", "cannot restore")
            root.manual_values["legacy"] = placeholder
            root._clean_constant_baseline = {"legacy": pickle.dumps(placeholder)}  # pyright: ignore[reportAttributeAccessIssue]
            with self.assertRaisesRegex(RegistrationError, "unrestorable legacy baseline"):
                root.run_all(clean_start=True)
            self.assertIs(root.manual_values["legacy"], placeholder)
            root.set_constant_value("legacy", [1])
            root.snapshot_constants()
            root.run_all(clean_start=True)
            self.assertEqual(root.manual_values["legacy"], [1])


class PickleOnlyConfigValue:
    def __init__(self, value: int) -> None:
        self.value = value

    def __deepcopy__(self, memo: dict[int, object]) -> PickleOnlyConfigValue:
        raise TypeError("deepcopy unsupported")

    def __reduce__(self) -> tuple[object, tuple[int]]:
        return (PickleOnlyConfigValue, (self.value,))


class CleanBaselineDiskTests(unittest.TestCase):
    def test_pickle_only_config_preserves_constructor_compatibility(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {"item": PickleOnlyConfigValue(1)}, Path(tmp) / "root")
            root.config["item"] = PickleOnlyConfigValue(5)
            root.run_all(clean_start=True)
            self.assertEqual(root.config["item"].value, 1)

    def test_disk_backed_snapshot_copies_artifact_without_loading(self) -> None:
        from mlpipelineholder.persistence.artifacts.store import ArtifactStore

        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {}, Path(tmp) / "root")
            root.set_constant_value("data", b"x" * 2_000_000, to_disk=True)
            root.add_block("draw", 1).register_function(optional_draw, ["result"])
            with patch.object(ArtifactStore, "load", side_effect=AssertionError("loaded baseline")):
                root.run_all()
            entry = (root._clean_constant_baseline or {})["data"]
            self.assertEqual(entry["kind"], "artifact")
            baseline = root.project_root / "clean_baseline" / entry["file"]
            self.assertTrue(baseline.is_file())
            self.assertEqual(baseline.stat().st_size, Path(root.manual_values["data"].file_path).stat().st_size)
            self.assertLess(len(pickle.dumps(root._serialize_payload())), 100_000)

    def test_staged_copy_failure_leaves_active_state_and_cleans_staging(self) -> None:
        from mlpipelineholder.persistence.artifacts.store import ArtifactStore

        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {"value": 1}, Path(tmp) / "root")
            root.set_constant_value("a", [1], to_disk=True)
            root.set_constant_value("b", [2], to_disk=True)
            root.add_block("draw", 1).register_function(optional_draw, ["result"])
            root.run_all()
            before = root.get_value("result")
            before_a = root.manual_values["a"]
            root.set_config("value", 9)
            copy_method = root.artifact_store.copy_artifact_file
            calls = 0

            def fail_second(*args: object, **kwargs: object) -> Path:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("copy failed")
                return copy_method(*args, **kwargs)  # type: ignore[arg-type]

            with patch.object(ArtifactStore, "copy_artifact_file", side_effect=fail_second):
                with self.assertRaisesRegex(OSError, "copy failed"):
                    root.run_all(clean_start=True)
            self.assertEqual(root.get_value("result"), before)
            self.assertIs(root.manual_values["a"], before_a)
            self.assertEqual(root.get_config_value("value"), 9)
            self.assertFalse(list((root.project_root / "artifacts" / "clean_baseline").glob("*")))
            self.assertIsNone(root._gate_cleanup_predecided)

    def test_failed_manual_tree_snapshot_keeps_every_previous_file(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {}, Path(tmp) / "root")
            child = PipelineHolder("child", {}, Path(tmp) / "child")
            root.set_constant_value("base", [1])
            child.set_constant_value("child_base", [2])
            root.add_child_pipeline(child, 1)
            root.snapshot_constants()
            before_root = dict(root._clean_constant_baseline or {})
            before_child = dict(child._clean_constant_baseline or {})
            root_file = root.project_root / "clean_baseline" / before_root["base"]["file"]
            child_file = child.project_root / "clean_baseline" / before_child["child_base"]["file"]
            original_root_bytes, original_child_bytes = root_file.read_bytes(), child_file.read_bytes()
            root.set_constant_value("base", [9])
            child.set_constant_value("child_base", [8])
            with patch.object(child, "_write_baseline_entry", side_effect=OSError("later child failed")):
                with self.assertRaisesRegex(RegistrationError, "later child failed"):
                    root.snapshot_constants()
            self.assertEqual(root._clean_constant_baseline, before_root)
            self.assertEqual(child._clean_constant_baseline, before_child)
            self.assertEqual(root_file.read_bytes(), original_root_bytes)
            self.assertEqual(child_file.read_bytes(), original_child_bytes)
            root.run_all(clean_start=True)
            self.assertEqual(root.get_constant_value("base"), [1])
            self.assertEqual(child.get_constant_value("child_base"), [2])

    def test_commit_failure_restores_outputs_config_and_artifact(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {"marker": 1}, Path(tmp) / "root")
            root.set_constant_value("items", [1], to_disk=True)
            block = root.add_block("count", 1)
            block.register_function(read_items, ["count"], save_to_disk=["count"])
            root.run_all()
            output_record = root.producer_outputs["count"]["count"]
            output_file = Path(output_record.file_path)
            root.set_config("marker", 9)
            original_invalidate = root._invalidate_all_outputs

            def fail_after_invalidation() -> None:
                original_invalidate()
                raise RuntimeError("commit failed")

            with patch.object(root, "_invalidate_all_outputs", side_effect=fail_after_invalidation):
                with self.assertRaisesRegex(RuntimeError, "commit failed"):
                    root.run_all(clean_start=True)
            self.assertTrue(output_file.exists())
            self.assertIs(root.producer_outputs["count"]["count"], output_record)
            self.assertEqual(root.get_value("count"), 1)
            self.assertEqual(root.get_config_value("marker"), 9)
            self.assertEqual(root.get_constant_value("items"), [1])
            self.assertIsNone(root._gate_cleanup_predecided)


class CleanBaselineLifecycleTests(unittest.TestCase):
    def test_publish_failure_rolls_back_all_pipeline_baselines(self) -> None:
        import os

        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {}, Path(tmp) / "root")
            child = PipelineHolder("child", {}, Path(tmp) / "child")
            root.set_constant_value("base", [1])
            child.set_constant_value("child_base", [2])
            root.add_child_pipeline(child, 1)
            root.snapshot_constants()
            root_entry = (root._clean_constant_baseline or {})["base"]
            child_entry = (child._clean_constant_baseline or {})["child_base"]
            root_file = root.project_root / "clean_baseline" / root_entry["file"]
            child_file = child.project_root / "clean_baseline" / child_entry["file"]
            previous_root, previous_child = root_file.read_bytes(), child_file.read_bytes()
            root.set_constant_value("base", [9])
            child.set_constant_value("child_base", [8])
            original_replace = os.replace

            def fail_child_publish(source: Path, destination: Path) -> None:
                if source.parent == child.metadata_root and source.name.startswith(".clean-baseline-stage-") and destination == child._clean_baseline_root:
                    raise OSError("promotion failed")
                original_replace(source, destination)

            with patch("mlpipelineholder.core.reproducibility.os.replace", side_effect=fail_child_publish):
                with self.assertRaisesRegex(OSError, "promotion failed"):
                    root.snapshot_constants()
            self.assertEqual(root_file.read_bytes(), previous_root)
            self.assertEqual(child_file.read_bytes(), previous_child)
            root.run_all(clean_start=True)
            self.assertEqual(root.manual_values["base"], [1])
            self.assertEqual(child.manual_values["child_base"], [2])

    def test_saved_baseline_files_relocate_and_outdated_bundle_files_clean_up(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {}, Path(tmp) / "root")
            root.set_constant_value("first", [1], to_disk=True)
            root.set_constant_value("second", [2])
            root.run_all()
            bundle = Path(tmp) / "bundle"
            root.save_pipeline(bundle)
            old_file = bundle / "clean_baseline" / (root._clean_constant_baseline or {})["second"]["file"]
            self.assertTrue(old_file.exists())
            root.manual_values.pop("second")
            root.snapshot_constants()
            root.save_pipeline(bundle, cleanup="auto")
            self.assertFalse(old_file.exists())
            loaded = PipelineHolder.load_pipeline(bundle, forced_deleting=True, trust_project=True)
            loaded.set_constant_value("first", [9], to_disk=True)
            loaded.run_all(clean_start=True)
            self.assertEqual(loaded.get_constant_value("first"), [1])
            self.assertIsInstance(loaded.manual_values["first"], ArtifactRecord)

    def test_prior_torch_warn_only_setting_survives_seeded_run(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed")
        previous_mode = torch.are_deterministic_algorithms_enabled()
        previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            with TemporaryDirectory() as tmp:
                root = PipelineHolder("root", {}, Path(tmp) / "root")
                root.add_block("draw", 1).register_function(torch_draw, ["value"])
                root.set_seed(7, deterministic_torch=True)
                root.run_all(clean_start=True)
                self.assertTrue(torch.are_deterministic_algorithms_enabled())
                self.assertTrue(torch.is_deterministic_algorithms_warn_only_enabled())
        finally:
            torch.use_deterministic_algorithms(previous_mode, warn_only=previous_warn_only)

    def test_optuna_sampler_resumes_at_identical_next_trial(self) -> None:
        try:
            import optuna
        except ImportError:
            self.skipTest("Optuna not installed")
        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {}, Path(tmp) / "root")
            study = optuna.create_study(study_name="sampler_baseline", sampler=optuna.samplers.RandomSampler(seed=17))
            objective = lambda trial: trial.suggest_float("x", 0, 1)
            study.optimize(objective, n_trials=1)
            root.set_constant_value("study", study)
            root.run_all()
            current = root.get_constant_value("study")
            current.optimize(objective, n_trials=1)
            expected = current.trials[1].params["x"]
            root.run_all(clean_start=True)
            restored = root.get_constant_value("study")
            restored.optimize(objective, n_trials=1)
            self.assertEqual(restored.trials[1].params["x"], expected)


class CleanBaselineFailureTests(unittest.TestCase):
    def test_sibling_capture_failure_keeps_ordinary_run_usable(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {}, Path(tmp) / "root")
            child = PipelineHolder("child", {}, Path(tmp) / "child")
            root.set_constant_value("valid", [1])
            child.set_constant_value("unrestorable", Lock(), copy=False)
            root.add_block("draw", 1).register_function(optional_draw, ["value"])
            root.add_child_pipeline(child, 2)
            root.run_all()
            self.assertIsNotNone(root.get_value("value"))
            self.assertIsNone(root._clean_constant_baseline)
            self.assertIsNone(child._clean_constant_baseline)
            self.assertFalse(list(root.metadata_root.glob(".clean-baseline-stage-*")))
            with self.assertRaisesRegex(RegistrationError, "Clean baseline unavailable"):
                root.run_all(clean_start=True)
            child.set_constant_value("unrestorable", [2])
            root.snapshot_constants()
            root.run_all(clean_start=True)
            self.assertEqual(child.get_constant_value("unrestorable"), [2])

    def test_child_commit_failure_restores_parent_mirror_and_sibling(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {}, Path(tmp) / "root")
            child = PipelineHolder("child", {}, Path(tmp) / "child")
            sibling = PipelineHolder("sibling", {}, Path(tmp) / "sibling")
            child.set_constant_value("items", [1])
            child.add_block("count", 1).register_function(read_items, ["child_result"])
            sibling.add_block("draw", 1).register_function(optional_draw, ["sibling_result"])
            root.add_child_pipeline(child, 1)
            root.add_child_pipeline(sibling, 2)
            root.run_all()
            prior_child = child.get_value("child_result")
            prior_sibling = sibling.get_value("sibling_result")
            prior_mirror = dict(root.producer_outputs.get("child", {}))
            real_invalidate = child._invalidate_all_outputs

            def fail_after_clear() -> None:
                real_invalidate()
                raise RuntimeError("child clear failed")

            with patch.object(child, "_invalidate_all_outputs", side_effect=fail_after_clear):
                with self.assertRaisesRegex(RuntimeError, "child clear failed"):
                    child.run_all(clean_start=True)
            self.assertEqual(child.get_value("child_result"), prior_child)
            self.assertEqual(sibling.get_value("sibling_result"), prior_sibling)
            self.assertEqual(root.producer_outputs.get("child", {}), prior_mirror)
            self.assertIsNone(child._gate_cleanup_predecided)

    def test_loaded_legacy_byte_baseline_does_not_silently_recapture(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {}, Path(tmp) / "root")
            root.set_constant_value("items", [1])
            root.run_all()
            root._clean_constant_baseline = {"items": pickle.dumps([1])}  # pyright: ignore[reportAttributeAccessIssue]
            root.set_constant_value("items", [9])
            root.save_pipeline(Path(tmp) / "bundle")
            loaded = PipelineHolder.load_pipeline(Path(tmp) / "bundle", forced_deleting=True, trust_project=True)
            with self.assertRaisesRegex(RegistrationError, "unrestorable legacy baseline"):
                loaded.run_all(clean_start=True)
            self.assertEqual(loaded.get_constant_value("items"), [9])


class CleanBaselineLazyInputTests(unittest.TestCase):
    def test_in_memory_dask_graph_snapshots_underlying_external_files(self) -> None:
        import shutil
        try:
            import dask.dataframe as dd
            import pandas as pd
        except ImportError:
            self.skipTest("Dask DataFrame and pandas not installed")
        with TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "external"
            dd.from_pandas(pd.DataFrame({"score": [1, 2, 3]}), npartitions=2).to_parquet(source_path)
            lazy = dd.read_parquet(source_path)
            root = PipelineHolder("root", {}, Path(tmp) / "root")
            root.set_constant_value("frame", lazy, copy=False)
            root.snapshot_constants()
            manifest = (root._clean_constant_baseline or {})["frame"]
            self.assertEqual(manifest["kind"], "dask")
            shutil.rmtree(source_path)
            root.run_all(clean_start=True)
            restored = root.get_constant_value("frame")
            self.assertIsInstance(restored, dd.DataFrame)
            self.assertEqual(restored.compute()["score"].tolist(), [1, 2, 3])


class CleanBaselineChildIsolationTests(unittest.TestCase):
    def test_child_clean_run_ignores_unavailable_parent_baseline(self) -> None:
        with TemporaryDirectory() as tmp:
            child = PipelineHolder("child", {}, Path(tmp) / "child")
            child.set_constant_value("items", [1])
            child.add_block("count", 1).register_function(read_items, ["child_result"])
            child.snapshot_constants()
            root = PipelineHolder("root", {}, Path(tmp) / "root")
            root.set_constant_value("uncopyable", Lock(), copy=False)
            root.add_child_pipeline(child, 1)
            root.run_all()
            self.assertIsNotNone(root._clean_baseline_error)
            child.set_constant_value("items", [9])
            child.run_all(clean_start=True)
            self.assertEqual(child.get_value("child_result"), 1)
            self.assertEqual(child.get_constant_value("items"), [1])
            self.assertIsNotNone(root._clean_baseline_error)
