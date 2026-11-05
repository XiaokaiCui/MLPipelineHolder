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

    def test_reject_unrestorable_constant_before_mutation(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {}, Path(tmp) / "root")
            with self.assertRaisesRegex(RegistrationError, "cannot be independently snapshotted"):
                root.set_constant_value("bad", Lock(), copy=False)
            self.assertNotIn("bad", root.manual_values)

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
    def test_constant_name_cannot_conflict_with_original_config_after_config_edit(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {"value": 1}, Path(tmp) / "root")
            root.config.pop("value")
            with self.assertRaisesRegex(RegistrationError, "definition-time configuration"):
                root.set_constant_value("value", 2)
            self.assertNotIn("value", root.manual_values)


class CleanStartLegacyBaselineTests(unittest.TestCase):
    def test_legacy_placeholder_requires_replacement_before_clean_start(self) -> None:
        with TemporaryDirectory() as tmp:
            root = PipelineHolder("root", {}, Path(tmp) / "root")
            placeholder = RuntimeValueReference("Lock", "<lock>", "cannot restore")
            root.manual_values["legacy"] = placeholder
            root._clean_constant_baseline = {"legacy": pickle.dumps(placeholder)}
            with self.assertRaisesRegex(RegistrationError, "unrestorable legacy baseline"):
                root.run_all(clean_start=True)
            self.assertIs(root.manual_values["legacy"], placeholder)
            root.set_constant_value("legacy", [1])
            root.snapshot_constants()
            root.run_all(clean_start=True)
            self.assertEqual(root.manual_values["legacy"], [1])
