"""Pandas and Dask DataFrame persistence, loading, and copy staging."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Final
import warnings

if TYPE_CHECKING:
    from contextlib import ExitStack

_DASK_TARGET_PARTITION_SIZE: Final = "256MiB"


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
                return dd.read_parquet(parquet_path)
        except Exception:
            pass
        import pandas as pd  # type: ignore

        return pd.read_parquet(path)
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
