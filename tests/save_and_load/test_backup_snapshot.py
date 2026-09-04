from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import import_module
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from typing import Callable, Protocol, cast
import unittest

from src.mlpipelineholder import PersistenceError, PipelineHandler, ResolutionError
from src.mlpipelineholder.models import ArtifactRecord


class BackupSnapshotProtocol(Protocol):
    def payload_for_path(self, path_parts: tuple[str, ...]) -> dict[str, object]: ...

    def assert_unchanged(self) -> None: ...


class BlockProtocol(Protocol):
    def register_function(
        self,
        function: Callable[..., object],
        output_variable_names: list[str],
    ) -> object: ...


ReadBackupSnapshot = Callable[[PipelineHandler], BackupSnapshotProtocol]

read_backup_snapshot = cast(
    ReadBackupSnapshot,
    import_module("src.mlpipelineholder.backup_snapshot").read_backup_snapshot,
)


@dataclass
class SnapshotConfig:
    base: int


def produce_seed(base: int) -> int:
    return base + 1


def child_total(seed: int, base: int) -> int:
    return seed + base


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_pickle(path: Path, value: object) -> None:
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def load_mapping(path: Path) -> dict[str, object]:
    return cast(dict[str, object], pickle.loads(path.read_bytes()))


def build_saved_root(temp_dir: str) -> tuple[PipelineHandler, Path, Path]:
    tmp_path = Path(temp_dir)
    work_root = tmp_path / "work"
    backup_root = tmp_path / "backup"
    root = PipelineHandler(
        "root",
        SnapshotConfig(base=2),
        work_root,
        pipeline_backup_directory=backup_root,
    )
    setup = cast(BlockProtocol, root.add_block("setup", 1))
    _ = setup.register_function(produce_seed, ["seed"])
    child = PipelineHandler("child", SnapshotConfig(base=5), tmp_path / "child")
    child_block = cast(BlockProtocol, child.add_block("child_block", 1))
    _ = child_block.register_function(child_total, ["child_total"])
    root.add_child_pipeline(child, 2)
    _ = root.run_all()
    _ = root.save_pipeline()
    return root, work_root, backup_root


class BackupSnapshotCharacterizationTests(unittest.TestCase):
    def test_in_place_save_creates_readable_backup_state_and_inspection_keeps_hashes(self) -> None:
        with TemporaryDirectory() as temp_dir:
            _, work_root, backup_root = build_saved_root(temp_dir)

            state_path = backup_root / "pipeline_state.pkl"
            config_path = backup_root / "config.pkl"
            meta_path = backup_root / "pipeline_meta.pkl"
            before_hashes = {
                "pipeline_state.pkl": file_sha256(state_path),
                "config.pkl": file_sha256(config_path),
                "pipeline_meta.pkl": file_sha256(meta_path),
            }

            state_payload = load_mapping(state_path)
            config_payload = load_mapping(config_path)
            meta_payload = load_mapping(meta_path)
            state_nodes = cast(list[dict[str, object]], state_payload["nodes"])
            child_payload = cast(dict[str, object], state_nodes[1]["payload"])
            config_data = cast(dict[str, object], config_payload["data"])

            self.assertEqual(state_payload["registration_name"], "root")
            self.assertEqual(state_payload["saved_project_root"], str(work_root))
            self.assertEqual(state_payload["pipeline_backup_directory"], str(backup_root))
            self.assertEqual(state_nodes[1]["registration_name"], "child")
            self.assertEqual(child_payload["registration_name"], "child")
            self.assertEqual(meta_payload["pipeline_directory"], str(work_root))
            self.assertEqual(meta_payload["pipeline_backup_directory"], str(backup_root))
            self.assertTrue(config_payload["__pipeline_serialized_config__"])
            self.assertEqual(config_payload["class_name"], "SnapshotConfig")
            self.assertEqual(config_data["base"], 2)

            after_hashes = {
                "pipeline_state.pkl": file_sha256(state_path),
                "config.pkl": file_sha256(config_path),
                "pipeline_meta.pkl": file_sha256(meta_path),
            }
            self.assertEqual(after_hashes, before_hashes)


class BackupSnapshotTests(unittest.TestCase):
    def test_read_backup_snapshot_returns_exact_nested_payload_and_preserves_hashes(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root, work_root, backup_root = build_saved_root(temp_dir)
            before_hashes = {
                name: file_sha256(backup_root / name)
                for name in ("pipeline_state.pkl", "config.pkl", "pipeline_meta.pkl")
            }

            snapshot = read_backup_snapshot(root)

            root_payload = snapshot.payload_for_path(("root",))
            child_payload = snapshot.payload_for_path(("root", "child"))
            self.assertEqual(root_payload["saved_project_root"], str(work_root))
            self.assertEqual(child_payload["registration_name"], "child")
            snapshot.assert_unchanged()

            after_hashes = {
                name: file_sha256(backup_root / name)
                for name in ("pipeline_state.pkl", "config.pkl", "pipeline_meta.pkl")
            }
            self.assertEqual(after_hashes, before_hashes)

    def test_read_backup_snapshot_requires_configured_existing_backup(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            root = PipelineHandler("root", SnapshotConfig(base=2), tmp_path / "work")

            with self.assertRaises(PersistenceError):
                _ = read_backup_snapshot(root)

        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            root = PipelineHandler(
                "root",
                SnapshotConfig(base=2),
                tmp_path / "work",
                pipeline_backup_directory=tmp_path / "missing-backup",
            )

            with self.assertRaises(PersistenceError):
                _ = read_backup_snapshot(root)

    def test_read_backup_snapshot_requires_all_current_format_files(self) -> None:
        for missing_name in ("pipeline_state.pkl", "config.pkl", "pipeline_meta.pkl"):
            with self.subTest(missing_name=missing_name):
                with TemporaryDirectory() as temp_dir:
                    root, _, backup_root = build_saved_root(temp_dir)
                    (backup_root / missing_name).unlink()

                    with self.assertRaises(PersistenceError):
                        _ = read_backup_snapshot(root)

    def test_read_backup_snapshot_rejects_corrupt_or_malformed_payloads(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root, _, backup_root = build_saved_root(temp_dir)
            _ = (backup_root / "pipeline_state.pkl").write_bytes(b"not-a-pickle")

            with self.assertRaises(PersistenceError):
                _ = read_backup_snapshot(root)

        with TemporaryDirectory() as temp_dir:
            root, _, backup_root = build_saved_root(temp_dir)
            write_pickle(backup_root / "pipeline_state.pkl", {"registration_name": "root"})

            with self.assertRaises(PersistenceError):
                _ = read_backup_snapshot(root)

        with TemporaryDirectory() as temp_dir:
            root, _, backup_root = build_saved_root(temp_dir)
            payload = load_mapping(backup_root / "pipeline_state.pkl")
            payload_nodes = cast(list[dict[str, object]], payload["nodes"])
            payload_nodes[1]["payload"] = []
            write_pickle(backup_root / "pipeline_state.pkl", payload)

            with self.assertRaises(PersistenceError):
                _ = read_backup_snapshot(root)

    def test_read_backup_snapshot_rejects_mismatched_registration_and_metadata(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root, work_root, backup_root = build_saved_root(temp_dir)
            payload = load_mapping(backup_root / "pipeline_state.pkl")
            payload["registration_name"] = "other-root"
            write_pickle(backup_root / "pipeline_state.pkl", payload)

            with self.assertRaises(PersistenceError):
                _ = read_backup_snapshot(root)

        with TemporaryDirectory() as temp_dir:
            root, _, backup_root = build_saved_root(temp_dir)
            payload = load_mapping(backup_root / "pipeline_state.pkl")
            payload["saved_project_root"] = "/somewhere/else"
            write_pickle(backup_root / "pipeline_state.pkl", payload)

            with self.assertRaises(PersistenceError):
                _ = read_backup_snapshot(root)

        with TemporaryDirectory() as temp_dir:
            root, work_root, backup_root = build_saved_root(temp_dir)
            write_pickle(
                backup_root / "pipeline_meta.pkl",
                {
                    "pipeline_directory": str(work_root / "other"),
                    "pipeline_backup_directory": str(backup_root),
                },
            )

            with self.assertRaises(PersistenceError):
                _ = read_backup_snapshot(root)

    def test_payload_for_path_requires_exact_rooted_hierarchy(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root, _, _ = build_saved_root(temp_dir)
            snapshot = read_backup_snapshot(root)

            with self.assertRaises(ResolutionError):
                _ = snapshot.payload_for_path(("child",))

            with self.assertRaises(ResolutionError):
                _ = snapshot.payload_for_path(("root", "missing"))

    def test_assert_unchanged_detects_fingerprint_drift(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root, _, backup_root = build_saved_root(temp_dir)
            snapshot = read_backup_snapshot(root)
            payload = load_mapping(backup_root / "pipeline_state.pkl")
            payload["run_history"] = [{"status": "drifted"}]
            write_pickle(backup_root / "pipeline_state.pkl", payload)

            with self.assertRaises(PersistenceError):
                snapshot.assert_unchanged()

    def test_read_backup_snapshot_ignores_missing_unrelated_artifact(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root, work_root, backup_root = build_saved_root(temp_dir)
            artifact_path = work_root / "artifacts" / "block" / "missing.json"
            payload = load_mapping(backup_root / "pipeline_state.pkl")
            payload["artifact_registry"] = {
                "unrelated": ArtifactRecord(
                    variable_name="unrelated",
                    serializer="json",
                    file_path=str(artifact_path),
                    produced_by_block="block",
                    produced_by_function="writer",
                    run_id="run",
                )
            }
            write_pickle(backup_root / "pipeline_state.pkl", payload)

            snapshot = read_backup_snapshot(root)
            missing = payload["artifact_registry"]["unrelated"]
            with self.assertRaises(PersistenceError):
                snapshot.validate_selected_artifacts(missing)

    def test_in_place_save_syncs_history_logs_into_backup(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            work_root = tmp_path / "work"
            backup_root = tmp_path / "backup"
            root = PipelineHandler(
                "root",
                SnapshotConfig(base=2),
                work_root,
                pipeline_backup_directory=backup_root,
            )
            setup = cast(BlockProtocol, root.add_block("setup", 1))
            _ = setup.register_function(produce_seed, ["seed"])
            _ = root.run_all()
            _ = root.save_pipeline()
            _ = root.save_pipeline()

            work_snapshots = sorted(work_root.joinpath("history_logs").glob("*.log"))
            backup_snapshots = sorted(backup_root.joinpath("history_logs").glob("*.log"))
            self.assertEqual(len(work_snapshots), 2)
            self.assertEqual(
                [snapshot.name for snapshot in backup_snapshots],
                [snapshot.name for snapshot in work_snapshots],
            )


if __name__ == "__main__":
    _ = unittest.main()
