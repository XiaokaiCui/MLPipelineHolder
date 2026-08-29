from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.mlpipelineholder import PipelineHandler
from src.mlpipelineholder.models import ArtifactRecord


def produce_dict() -> dict[str, int]:
    return {"v": 1}


def _block_dir(project_root: Path) -> Path:
    for artifacts_dir in project_root.rglob("artifacts"):
        for block_dir in artifacts_dir.iterdir():
            if block_dir.is_dir():
                return block_dir
    raise AssertionError("no artifact block directory found")


def _generation_entries(project_root: Path) -> list[Path]:
    entries: list[Path] = []
    for artifacts_dir in project_root.rglob("artifacts"):
        for block_dir in artifacts_dir.iterdir():
            if block_dir.is_dir():
                entries.extend(block_dir.iterdir())
    return sorted(entries)


def _add_stale_generation(project_root: Path) -> Path:
    stale = _block_dir(project_root) / (
        "ghost_output__value__00000000000000000000000000000001"
        "__00000000000000000000000000000002.json"
    )
    stale.write_text('{"x": 1}')
    return stale


def _saved_root(temp_dir: str, *, backup_directory: Path | None = None) -> PipelineHandler:
    root_path = Path(temp_dir)
    pipeline = PipelineHandler("root", {}, root_path / "root", pipeline_backup_directory=backup_directory)
    block = pipeline.add_block("produce", 1)
    if block is None:
        raise AssertionError("add_block should return a block")
    block.register_function(produce_dict, ["data"], save_to_disk=["data"])
    return pipeline


def _add_ghost_child_tree(root: Path, name: str) -> Path:
    """Create a fully framework-managed orphaned child pipeline tree."""
    ghost = root / "children" / name
    ghost.mkdir(parents=True)
    (ghost / "pipeline_state.pkl").write_bytes(b"x")
    (ghost / "config.pkl").write_bytes(b"x")
    (ghost / "pipeline_meta.pkl").write_bytes(b"x")
    (ghost / "metadata").mkdir()
    (ghost / "metadata" / "pipeline.log").write_text("log")
    (ghost / "history_logs").mkdir()
    (ghost / "artifacts").mkdir()
    return ghost


def _artifact_record(path: Path, serializer: str) -> ArtifactRecord:
    return ArtifactRecord(
        variable_name="manual_artifact",
        serializer=serializer,
        file_path=str(path),
        produced_by_block="manual",
        produced_by_function="manual",
        run_id="manual",
    )


class SaveCleanupTests(unittest.TestCase):
    def test_auto_cleanup_deletes_unreferenced_generations(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "root"
            pipeline = _saved_root(temp_dir)
            pipeline.run_all()
            pipeline.save_pipeline()
            stale = _add_stale_generation(root)
            self.assertTrue(stale.exists())

            pipeline.save_pipeline()

            self.assertFalse(stale.exists())
            self.assertEqual(len(_generation_entries(root)), 1)
            loaded = PipelineHandler.load_pipeline(root, forced_deleting=True)
            self.assertEqual(loaded.get_value("data"), {"v": 1})

    def test_none_cleanup_keeps_everything(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "root"
            pipeline = _saved_root(temp_dir)
            pipeline.run_all()
            pipeline.save_pipeline()
            stale = _add_stale_generation(root)
            self.assertTrue(stale.exists())

            pipeline.save_pipeline(cleanup="none")

            self.assertTrue(stale.exists())
            self.assertEqual(len(_generation_entries(root)), 2)

    def test_confirm_cleanup_deletes_only_after_yes(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "root"
            pipeline = _saved_root(temp_dir)
            pipeline.run_all()
            pipeline.save_pipeline()
            stale = _add_stale_generation(root)

            with patch("builtins.input", return_value="yes"):
                pipeline.save_pipeline(cleanup="confirm")

            self.assertFalse(stale.exists())
            self.assertEqual(len(_generation_entries(root)), 1)

    def test_confirm_cleanup_keeps_after_no(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "root"
            pipeline = _saved_root(temp_dir)
            pipeline.run_all()
            pipeline.save_pipeline()
            stale = _add_stale_generation(root)

            with patch("builtins.input", return_value="n"):
                pipeline.save_pipeline(cleanup="confirm")

            self.assertTrue(stale.exists())
            self.assertEqual(len(_generation_entries(root)), 2)

    def test_confirm_cleanup_keeps_after_eof(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "root"
            pipeline = _saved_root(temp_dir)
            pipeline.run_all()
            pipeline.save_pipeline()
            stale = _add_stale_generation(root)

            with patch("builtins.input", side_effect=EOFError):
                pipeline.save_pipeline(cleanup="confirm")

            self.assertTrue(stale.exists())
            self.assertEqual(len(_generation_entries(root)), 2)

    def test_invalid_cleanup_mode_raises_before_saving(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "root"
            pipeline = _saved_root(temp_dir)
            pipeline.run_all()

            with self.assertRaises(ValueError):
                pipeline.save_pipeline(cleanup="bogus")

            self.assertFalse((root / "pipeline_state.pkl").exists())

    def test_auto_cleanup_deletes_orphaned_managed_child_tree(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "root"
            pipeline = _saved_root(temp_dir)
            pipeline.run_all()
            pipeline.save_pipeline()
            ghost = _add_ghost_child_tree(root, "ghost_a")
            self.assertTrue(ghost.is_dir())

            pipeline.save_pipeline()

            self.assertFalse(ghost.exists())

    def test_orphaned_child_with_unknown_file_is_protected(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "root"
            pipeline = _saved_root(temp_dir)
            pipeline.run_all()
            pipeline.save_pipeline()
            ghost = _add_ghost_child_tree(root, "ghost_a")
            (ghost / "notes.txt").write_text("user data")
            ghost_artifacts = ghost / "artifacts" / "ghost_block"
            ghost_artifacts.mkdir()
            stale_in_ghost = ghost_artifacts / (
                "ghost_output__value__00000000000000000000000000000001"
                "__00000000000000000000000000000002.json"
            )
            stale_in_ghost.write_text('{"x": 1}')

            pipeline.save_pipeline()

            self.assertTrue(ghost.exists())
            self.assertTrue((ghost / "notes.txt").exists())
            self.assertTrue(stale_in_ghost.exists())

    def test_orphaned_child_containing_live_artifact_is_protected(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "root"
            pipeline = _saved_root(temp_dir)
            pipeline.run_all()
            pipeline.save_pipeline(cleanup="none")
            ghost = _add_ghost_child_tree(root, "ghost_a")
            live_artifact = ghost / "artifacts" / "manual" / (
                "manual__live__manual__00000000000000000000000000000001.json"
            )
            live_artifact.parent.mkdir()
            live_artifact.write_text('{"keep": true}')
            pipeline.set_constant_value(
                "live_manual",
                _artifact_record(live_artifact, "json"),
                copy=False,
            )

            pipeline.save_pipeline()

            self.assertTrue(ghost.exists())
            self.assertTrue(live_artifact.exists())
            loaded = PipelineHandler.load_pipeline(root, forced_deleting=True)
            self.assertEqual(loaded.get_constant_value("live_manual"), {"keep": True})

    def test_cleanup_discovery_error_does_not_fail_committed_save(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "root"
            backup = Path(temp_dir) / "backup"
            pipeline = _saved_root(temp_dir, backup_directory=backup)
            pipeline.run_all()
            pipeline.save_pipeline(cleanup="none")
            pipeline.set_constant_value("saved_after_error", 2)
            (root / "refresh_marker.txt").write_text("new backup state")
            original_iterdir = Path.iterdir

            def fail_artifact_scan(path: Path):
                if path == root / "artifacts":
                    raise PermissionError("blocked artifact directory")
                return original_iterdir(path)

            with patch.object(Path, "iterdir", fail_artifact_scan):
                saved_path = pipeline.save_pipeline()

            self.assertEqual(saved_path, root)
            self.assertTrue((backup / "refresh_marker.txt").exists())
            loaded = PipelineHandler.load_pipeline(root, forced_deleting=True)
            self.assertEqual(loaded.get_constant_value("saved_after_error"), 2)

    def test_cleanup_ignores_non_framework_generation_names(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "root"
            pipeline = _saved_root(temp_dir)
            pipeline.run_all()
            pipeline.save_pipeline()
            block_dir = _block_dir(root)
            incomplete_name = block_dir / (
                "note__00000000000000000000000000000001.json"
            )
            unknown_suffix = block_dir / (
                "manual__value__run__00000000000000000000000000000002.txt"
            )
            incomplete_name.write_text("keep")
            unknown_suffix.write_text("keep")

            pipeline.save_pipeline()

            self.assertTrue(incomplete_name.exists())
            self.assertTrue(unknown_suffix.exists())

    def test_cleanup_does_not_scan_inside_live_directory_artifact(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "root"
            pipeline = _saved_root(temp_dir)
            pipeline.run_all()
            pipeline.save_pipeline(cleanup="none")
            live_directory = _block_dir(root) / (
                "manual__directory__run__00000000000000000000000000000001.parquet"
            )
            nested_generation = live_directory / "artifacts" / "nested" / (
                "manual__nested__run__00000000000000000000000000000002.json"
            )
            nested_generation.parent.mkdir(parents=True)
            nested_generation.write_text('{"keep": true}')
            pipeline.set_constant_value(
                "live_directory",
                _artifact_record(live_directory, "parquet"),
                copy=False,
            )

            pipeline.save_pipeline()

            self.assertTrue(live_directory.exists())
            self.assertTrue(nested_generation.exists())

    def test_alternate_target_save_does_not_clean_source(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            root = tmp / "root"
            pipeline = _saved_root(temp_dir)
            pipeline.run_all()
            pipeline.save_pipeline()
            stale = _add_stale_generation(root)
            self.assertTrue(stale.exists())

            pipeline.save_pipeline(tmp / "bundle")

            self.assertTrue(stale.exists())
            self.assertEqual(len(_generation_entries(root)), 2)


if __name__ == "__main__":
    unittest.main()
