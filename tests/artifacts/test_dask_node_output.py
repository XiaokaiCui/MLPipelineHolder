from __future__ import annotations

# pyright: basic
import unittest
from datetime import date, time
from decimal import Decimal
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from mlpipelineholder import PipelineHandler
from mlpipelineholder.core.models import ArtifactRecord
from mlpipelineholder.integrations.dataframe import dump_dataframe, load_dataframe


def produce_disk_frame():
    dd = import_module("dask.dataframe")
    pd = import_module("pandas")
    return dd.from_pandas(pd.DataFrame({"value": [0]}), npartitions=1)


def produce_memory_value() -> int:
    return 0


def produce_list_frame():
    dask = import_module("dask")
    dd = import_module("dask.dataframe")
    pd = import_module("pandas")
    with dask.config.set({"dataframe.convert-string": False}):
        return dd.from_pandas(
            pd.DataFrame(
                {
                    "ticker": ["A", "B"],
                    "weekly_return_list": [[0.1, 0.2], [0.3]],
                }
            ),
            npartitions=1,
        )


def produce_arrow_list_frame():
    dask = import_module("dask")
    dd = import_module("dask.dataframe")
    pd = import_module("pandas")
    pa = import_module("pyarrow")
    data = {
        "ticker": pd.array(["A", "B"], dtype="string[pyarrow]"),
        "python_string": pd.Series(["A", None], dtype="string[python]"),
        "Date": pd.Series(
            ["2025-01-03", "2025-01-10"],
            dtype="datetime64[ns]",
        ),
        "elapsed": pd.Series([1, 2], dtype="timedelta64[ns]"),
        "floating": pd.Series([1.5, 2.5], dtype="float32"),
        "nullable_float32": pd.Series([1.5, None], dtype="Float32"),
        "nullable_float64": pd.Series([1.5, None], dtype="Float64"),
        "flag": pd.Series([True, False], dtype="bool"),
        "nullable_flag": pd.Series([True, None], dtype="boolean"),
        "segment": pd.Series(
            ["large", "small"],
            dtype=pd.CategoricalDtype(ordered=True),
        ),
        "python_date": pd.Series([date(2025, 1, 3), None], dtype="object"),
        "python_time": pd.Series([time(12, 30), None], dtype="object"),
        "binary": pd.Series([b"A", None], dtype="object"),
        "decimal": pd.Series([Decimal("1.25"), None], dtype="object"),
        "arrow_integer": pd.array([1, None], dtype=pd.ArrowDtype(pa.int64())),
        "weekly_return_list": pd.array(
            [[0.1, 0.2], [0.3]],
            dtype=pd.ArrowDtype(pa.list_(pa.float64())),
        ),
        "arrow_struct": pd.array(
            [{"score": 1.5}, {"score": 2.5}],
            dtype=pd.ArrowDtype(pa.struct([("score", pa.float64())])),
        ),
        "arrow_map": pd.array(
            [[("score", 1.5)], [("score", 2.5)]],
            dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.float64())),
        ),
    }
    for dtype in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"):
        data[dtype] = pd.Series([1, 2], dtype=dtype)
    for dtype in ("Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64"):
        data[f"nullable_{dtype}"] = pd.Series([1, None], dtype=dtype)
    for unit in ("s", "ms", "us", "ns"):
        data[f"datetime_{unit}"] = pd.Series(
            ["2025-01-03", "2025-01-10"],
            dtype=f"datetime64[{unit}]",
        )
        data[f"datetime_utc_{unit}"] = pd.Series(
            ["2025-01-03", "2025-01-10"],
            dtype=pd.DatetimeTZDtype(unit=unit, tz="UTC"),
        )
        data[f"timedelta_{unit}"] = pd.Series([1, 2], dtype=f"timedelta64[{unit}]")
    with dask.config.set({"dataframe.convert-string": False}):
        return dd.from_pandas(pd.DataFrame(data), npartitions=1)


def produce_arrow_index_frame():
    dd = import_module("dask.dataframe")
    frame = produce_arrow_list_frame().compute().set_index("Date")
    return dd.from_pandas(frame, npartitions=1)


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
    def test_pandas_arrow_fallback_restores_declared_dtypes(self) -> None:
        pd = import_module("pandas")
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "frame.parquet"
            expected = produce_arrow_list_frame().compute()
            expected["period"] = pd.Series(pd.period_range("2025-01", periods=2, freq="M"))
            expected["interval"] = pd.Series(pd.arrays.IntervalArray.from_breaks([0, 1, 2]))
            expected.to_parquet(path)

            actual = load_dataframe("parquet", path)

            self.assertIsInstance(actual, pd.DataFrame)
            self.assertEqual(str(actual["Date"].dtype), "datetime64[ns]")
            self.assertEqual(str(actual["datetime_utc_ms"].dtype), "datetime64[ms, UTC]")
            self.assertEqual(str(actual["timedelta_us"].dtype), "timedelta64[us]")
            self.assertEqual(str(actual["nullable_UInt32"].dtype), "UInt32")
            self.assertEqual(str(actual["segment"].dtype), "category")
            self.assertEqual(str(actual["period"].dtype), "period[M]")
            self.assertEqual(str(actual["interval"].dtype), "interval[int64, right]")
            self.assertEqual(actual["weekly_return_list"].tolist(), [[0.1, 0.2], [0.3]])

            indexed_path = Path(temp_dir) / "indexed.parquet"
            expected.set_index(["Date", "int16"]).to_parquet(indexed_path)
            indexed = load_dataframe("parquet", indexed_path)
            self.assertEqual(str(indexed.index.levels[0].dtype), "datetime64[ns]")
            self.assertEqual(str(indexed.index.levels[1].dtype), "int16")

    def test_dask_arrow_fallback_restores_index_dtype(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("root", {}, Path(temp_dir) / "project")
            block = pipeline.add_block("producer", 1)
            if block is None:
                raise AssertionError("add_block should return a block")
            block.register_function(
                produce_arrow_index_frame,
                ["value"],
                save_to_disk=["value"],
            )
            _ = pipeline.run_all()

            projected = pipeline.get_node_output("producer", "value")[["ticker"]].compute()

            self.assertEqual(str(projected.index.dtype), "datetime64[ns]")
            self.assertEqual(projected["ticker"].tolist(), ["A", "B"])

    def test_arrow_fallback_restores_supported_pandas_dtypes(self) -> None:
        pd = import_module("pandas")
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("root", {}, Path(temp_dir) / "project")
            block = pipeline.add_block("producer", 1)
            if block is None:
                raise AssertionError("add_block should return a block")
            block.register_function(
                produce_arrow_list_frame,
                ["value"],
                save_to_disk=["value"],
            )
            _ = pipeline.run_all()

            lazy_value = pipeline.get_node_output("producer", "value")
            projected = lazy_value[["ticker", "Date"]].compute()
            actual = lazy_value.compute()

            self.assertEqual(projected["ticker"].tolist(), ["A", "B"])
            self.assertEqual(str(projected["Date"].dtype), "datetime64[ns]")
            self.assertEqual(actual["ticker"].tolist(), ["A", "B"])
            self.assertEqual(str(actual["python_string"].dtype), "string")
            self.assertEqual(str(actual["Date"].dtype), "datetime64[ns]")
            self.assertEqual(str(actual["elapsed"].dtype), "timedelta64[ns]")
            self.assertEqual(str(actual["floating"].dtype), "float32")
            self.assertEqual(str(actual["nullable_float32"].dtype), "Float32")
            self.assertEqual(str(actual["nullable_float64"].dtype), "Float64")
            self.assertEqual(str(actual["flag"].dtype), "bool")
            self.assertEqual(str(actual["nullable_flag"].dtype), "boolean")
            self.assertEqual(str(actual["segment"].dtype), "category")
            self.assertTrue(actual["segment"].cat.ordered)
            self.assertEqual(actual["python_date"].iloc[0], date(2025, 1, 3))
            self.assertEqual(actual["python_time"].iloc[0], time(12, 30))
            self.assertEqual(actual["binary"].iloc[0], b"A")
            self.assertEqual(actual["decimal"].iloc[0], Decimal("1.25"))
            for column in ("python_date", "python_time", "binary", "decimal"):
                self.assertTrue(pd.isna(actual[column].iloc[1]))
            self.assertEqual(str(actual["arrow_integer"].dtype), "int64[pyarrow]")
            self.assertEqual(actual["weekly_return_list"].tolist(), [[0.1, 0.2], [0.3]])
            self.assertEqual(actual["arrow_struct"].tolist(), [{"score": 1.5}, {"score": 2.5}])
            self.assertEqual(
                actual["arrow_map"].tolist(),
                [[("score", 1.5)], [("score", 2.5)]],
            )
            for dtype in (
                "int8",
                "int16",
                "int32",
                "int64",
                "uint8",
                "uint16",
                "uint32",
                "uint64",
            ):
                self.assertEqual(str(actual[dtype].dtype), dtype)
            for dtype in (
                "Int8",
                "Int16",
                "Int32",
                "Int64",
                "UInt8",
                "UInt16",
                "UInt32",
                "UInt64",
            ):
                self.assertEqual(str(actual[f"nullable_{dtype}"].dtype), dtype)
            for unit in ("s", "ms", "us", "ns"):
                self.assertTrue(pd.api.types.is_datetime64_dtype(actual[f"datetime_{unit}"].dtype))
                self.assertIsInstance(actual[f"datetime_utc_{unit}"].dtype, pd.DatetimeTZDtype)
                self.assertTrue(pd.api.types.is_timedelta64_dtype(actual[f"timedelta_{unit}"].dtype))

    def test_disk_backed_list_column_retries_without_global_arrow_schema(self) -> None:
        with TemporaryDirectory() as temp_dir:
            pipeline = PipelineHandler("root", {}, Path(temp_dir) / "project")
            block = pipeline.add_block("producer", 1)
            if block is None:
                raise AssertionError("add_block should return a block")
            block.register_function(
                produce_list_frame,
                ["value"],
                save_to_disk=["value"],
            )

            with self.assertWarnsRegex(UserWarning, "retrying with schema=None"):
                _ = pipeline.run_all()

            actual = pipeline.get_node_output("producer", "value").compute()
            self.assertEqual(actual["ticker"].tolist(), ["A", "B"])
            self.assertEqual(list(actual["weekly_return_list"].iloc[0]), [0.1, 0.2])
            self.assertEqual(list(actual["weekly_return_list"].iloc[1]), [0.3])

    def test_non_arrow_dask_parquet_failure_is_not_retried(self) -> None:
        frame = produce_list_frame()
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "frame.parquet"
            with patch.object(type(frame), "to_parquet", side_effect=OSError("disk full")) as save:
                with self.assertRaisesRegex(OSError, "disk full"):
                    dump_dataframe(frame, "parquet", path)

            save.assert_called_once_with(path)

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
