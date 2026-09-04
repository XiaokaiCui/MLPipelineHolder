from __future__ import annotations

# pyright: basic
import unittest
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory

from mlpipelineholder import PipelineHandler
from mlpipelineholder.models import ArtifactRecord


def produce_disk_frame():
    dd = import_module("dask.dataframe")
    pd = import_module("pandas")
    return dd.from_pandas(pd.DataFrame({"value": [0]}), npartitions=1)


def produce_memory_value() -> int:
    return 0


def limited_read_frame(read_count: list[int]):
    dask = import_module("dask")
    dd = import_module("dask.dataframe")
    pd = import_module("pandas")
    meta = pd.DataFrame({"value": pd.Series(dtype="int64")})

    @dask.delayed
    def load_partition():
        read_count[0] += 1
        if read_count[0] > 2:
            raise FileNotFoundError("source is no longer available")
        return pd.DataFrame({"value": [1, 2, 3]})

    return dd.from_delayed([load_partition()], meta=meta)


@unittest.skipUnless(
    find_spec("dask.dataframe") is not None,
    "dask.dataframe is not available",
)
class DaskNodeOutputTests(unittest.TestCase):
    def test_disk_backed_target_stages_dask_replacement_before_saving(self) -> None:
        # Given: a disk-backed target and a valid Dask replacement whose source
        # can be evaluated only twice.
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("root", {}, Path(temp_dir) / "project")
            block = pipeline.add_block("producer", 1)
            if block is None:
                raise AssertionError("add_block should return a block")
            block.register_function(
                produce_disk_frame,
                ["value"],
                save_to_disk=["value"],
            )
            _ = pipeline.run_all()
            read_count = [0]
            replacement = limited_read_frame(read_count)
            self.assertEqual(replacement.compute()["value"].tolist(), [1, 2, 3])

            # When: the Dask collection replaces the disk-backed output.
            pipeline.set_node_output("producer", "value", replacement)

            # Then: the replacement remains disk-backed and materializes as Dask.
            record = pipeline.producer_outputs["producer"]["value"]
            self.assertIsInstance(record, ArtifactRecord)
            actual = pipeline.get_node_output("producer", "value")
            self.assertEqual(actual.compute()["value"].tolist(), [1, 2, 3])

    def test_in_memory_target_keeps_dask_replacement_in_memory(self) -> None:
        # Given: an in-memory target and a lazy Dask replacement.
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("root", {}, Path(temp_dir) / "project")
            block = pipeline.add_block("producer", 1)
            if block is None:
                raise AssertionError("add_block should return a block")
            block.register_function(produce_memory_value, ["value"])
            _ = pipeline.run_all()
            read_count = [0]
            replacement = limited_read_frame(read_count)

            # When: the Dask collection replaces the in-memory output.
            pipeline.set_node_output("producer", "value", replacement)

            # Then: no disk artifact or eager evaluation is introduced.
            stored = pipeline.producer_outputs["producer"]["value"]
            self.assertNotIsInstance(stored, ArtifactRecord)
            self.assertEqual(read_count, [0])
            self.assertEqual(stored.compute()["value"].tolist(), [1, 2, 3])


if __name__ == "__main__":
    _ = unittest.main()
