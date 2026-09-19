"""Pandas and Dask DataFrame persistence, loading, and copy staging."""

from __future__ import annotations

import json
from importlib.util import find_spec
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Final, cast
import warnings

if TYPE_CHECKING:
    from contextlib import ExitStack

_DASK_TARGET_PARTITION_SIZE: Final = "256MiB"


def _is_unreadable_arrow_extension_dtype(exc: TypeError) -> bool:
    message = str(exc)
    return "data type" in message and "[pyarrow]" in message and "not understood" in message


def _read_pandas_parquet_metadata(path: Path) -> list[dict[str, Any]]:
    """Return pandas column metadata stored in a Parquet dataset."""
    import pyarrow.dataset as ds  # type: ignore

    schema_metadata = ds.dataset(path, format="parquet").schema.metadata or {}
    pandas_metadata = schema_metadata.get(b"pandas")
    if pandas_metadata is None:
        return []

    try:
        metadata = json.loads(pandas_metadata)
    except (AttributeError, json.JSONDecodeError, UnicodeDecodeError):
        return []
    if not isinstance(metadata, dict):
        return []
    columns = metadata.get("columns", [])
    if not isinstance(columns, list):
        return []
    return [cast(dict[str, Any], column) for column in columns if isinstance(column, dict)]


def _requires_arrow_dtype_backend(columns: list[dict[str, Any]]) -> bool:
    """Return whether pandas cannot reconstruct any declared Parquet dtype."""
    import pandas as pd  # type: ignore

    for column in columns:
        try:
            pd.api.types.pandas_dtype(column["numpy_type"])
        except (KeyError, TypeError, ValueError):
            return True
    return False


def _arrow_table_to_pandas(table: Any) -> Any:
    """Convert Arrow after neutralizing only unreadable pandas dtype metadata."""
    import pandas as pd  # type: ignore

    schema_metadata = dict(table.schema.metadata or {})
    pandas_metadata = schema_metadata.get(b"pandas")
    if pandas_metadata is None:
        return table.to_pandas()
    try:
        metadata = json.loads(pandas_metadata)
        columns = metadata.get("columns", [])
    except (AttributeError, json.JSONDecodeError, UnicodeDecodeError):
        return table.to_pandas()
    if not isinstance(metadata, dict) or not isinstance(columns, list):
        return table.to_pandas()

    arrow_backed_fields: set[str] = set()
    metadata_changed = False
    for column in columns:
        if not isinstance(column, dict):
            continue
        dtype_name = column.get("numpy_type")
        try:
            pd.api.types.pandas_dtype(dtype_name)
        except (KeyError, TypeError, ValueError):
            field_name = column.get("field_name")
            if (
                isinstance(field_name, str)
                and isinstance(dtype_name, str)
                and "[pyarrow]" in dtype_name
            ):
                arrow_backed_fields.add(field_name)
            column["numpy_type"] = "object"
            metadata_changed = True

    if metadata_changed:
        schema_metadata[b"pandas"] = json.dumps(metadata).encode()
        table = table.replace_schema_metadata(schema_metadata)
    value = table.to_pandas()

    arrow_index_fields = arrow_backed_fields.intersection(value.index.names)
    if arrow_index_fields:
        index_arrays = [
            pd.array(
                table.column(name),
                dtype=pd.ArrowDtype(table.schema.field(name).type),
            )
            if name in arrow_index_fields
            else value.index.get_level_values(name)
            for name in value.index.names
        ]
        value.index = (
            pd.Index(index_arrays[0], name=value.index.name)
            if len(index_arrays) == 1
            else pd.MultiIndex.from_arrays(index_arrays, names=value.index.names)
        )
    for field_name in arrow_backed_fields.intersection(value.columns):
        try:
            dtype = pd.ArrowDtype(table.schema.field(field_name).type)
            array = pd.array(table.column(field_name), dtype=dtype)
        except (KeyError, TypeError, ValueError):
            continue
        value[field_name] = pd.Series(array, index=value.index, name=field_name)
    return value


def _read_arrow_parquet_partition(path: str) -> Any:
    """Read one Parquet partition after sanitizing unreadable dtype metadata."""
    import pyarrow.parquet as pq  # type: ignore

    return _arrow_table_to_pandas(pq.read_table(path))


def _load_dask_arrow_parquet(path: Path) -> Any:
    """Build a projection-safe lazy Dask DataFrame from Arrow-backed partitions."""
    import dask  # type: ignore
    import dask.dataframe as dd  # type: ignore
    import pyarrow.dataset as ds  # type: ignore

    dataset = ds.dataset(path, format="parquet")
    meta = _arrow_table_to_pandas(dataset.schema.empty_table())
    with dask.config.set({"dataframe.convert-string": False}):
        return dd.from_map(
            _read_arrow_parquet_partition,
            dataset.files,
            meta=meta,
        )


def _load_pandas_arrow_parquet(path: Path) -> Any:
    """Read Parquet after sanitizing unreadable pandas dtype metadata."""
    import pyarrow.parquet as pq  # type: ignore

    return _arrow_table_to_pandas(pq.read_table(path))


def _dump_dask_parquet(value: Any, path: Path) -> None:
    """Write a Dask DataFrame, retrying when Arrow cannot infer object metadata."""
    import pyarrow as pa  # type: ignore

    try:
        value.to_parquet(path)
    except pa.ArrowInvalid:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
        warnings.warn(
            "Dask Parquet schema inference failed; retrying with schema=None. "
            "Each partition will infer its own schema, so partition schemas must remain compatible.",
            UserWarning,
            stacklevel=2,
        )
        value.to_parquet(path, schema=None)


def choose_dask_serializer(value: Any) -> str | None:
    """Return the serializer for a Dask DataFrame, or ``None`` for other values."""
    try:
        import dask.dataframe as dd  # type: ignore

        if isinstance(value, dd.DataFrame):
            return "parquet"
    except Exception:
        return None
    return None


def choose_pandas_serializer(value: Any) -> str | None:
    """Return the serializer for a pandas DataFrame, or ``None`` for other values."""
    try:
        import pandas as pd  # type: ignore

        if isinstance(value, pd.DataFrame):
            if len(value) > 3_000_000:
                return "parquet"
            if find_spec("pyarrow") is not None:
                return "feather"
            return "pickle"
    except Exception:
        return None
    return None


def dump_dataframe(value: Any, serializer: str, path: Path) -> None:
    """Write a DataFrame to disk using the feather or parquet serializer."""
    if serializer == "feather":
        import pyarrow as pa  # type: ignore
        import pyarrow.ipc as ipc  # type: ignore

        table = pa.Table.from_pandas(value, preserve_index=True)
        with path.open("wb") as handle:
            with ipc.new_file(handle, table.schema) as writer:
                writer.write_table(table)
        return
    if serializer == "parquet":
        try:
            import dask.dataframe as dd  # type: ignore
        except ImportError:
            dd = None
        if dd is not None and isinstance(value, dd.DataFrame):
            value = value.repartition(partition_size=_DASK_TARGET_PARTITION_SIZE)
            if value.npartitions == 1:
                value = value.repartition(npartitions=2)
            _dump_dask_parquet(value, path)
            return
        value.to_parquet(path)
        return
    raise ValueError(f"Unsupported dataframe serializer: {serializer}")


def load_dataframe(serializer: str, path: Path) -> Any:
    """Read a DataFrame from disk using the feather or parquet serializer."""
    if serializer == "feather":
        import pyarrow.ipc as ipc  # type: ignore

        with path.open("rb") as handle:
            table = ipc.open_file(handle).read_all()
        return table.to_pandas()
    if serializer == "parquet":
        try:
            import dask.dataframe as dd  # type: ignore

            parquet_path = Path(path)
            if parquet_path.is_dir():
                pandas_metadata = _read_pandas_parquet_metadata(parquet_path)
                if _requires_arrow_dtype_backend(pandas_metadata):
                    return _load_dask_arrow_parquet(parquet_path)
                try:
                    return dd.read_parquet(parquet_path)
                except TypeError as exc:
                    if not _is_unreadable_arrow_extension_dtype(exc):
                        raise
                    return _load_dask_arrow_parquet(parquet_path)
        except Exception:
            pass
        import pandas as pd  # type: ignore

        parquet_path = Path(path)
        pandas_metadata = _read_pandas_parquet_metadata(parquet_path)
        if _requires_arrow_dtype_backend(pandas_metadata):
            return _load_pandas_arrow_parquet(parquet_path)
        try:
            return pd.read_parquet(parquet_path)
        except TypeError as exc:
            if not _is_unreadable_arrow_extension_dtype(exc):
                raise
            return _load_pandas_arrow_parquet(parquet_path)
    raise ValueError(f"Unsupported dataframe serializer: {serializer}")


def stage_dask_dataframe_for_copy(
    value: Any,
    temp_parent: Path,
    stack: ExitStack,
) -> Any:
    """Snapshot a Dask DataFrame to a managed temporary parquet copy before saving."""
    try:
        import dask.dataframe as dd  # type: ignore
    except ModuleNotFoundError:
        return value
    if not isinstance(value, dd.DataFrame):
        return value
    temp_dir = stack.enter_context(
        TemporaryDirectory(prefix=".dask-replacement-", dir=temp_parent)
    )
    snapshot_path = Path(temp_dir) / "snapshot.parquet"
    _dump_dask_parquet(value, snapshot_path)
    return dd.read_parquet(snapshot_path)
