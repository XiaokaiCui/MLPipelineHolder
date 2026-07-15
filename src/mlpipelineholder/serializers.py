from __future__ import annotations

import json
import pickle
from importlib.util import find_spec
from pathlib import Path
from typing import Any


def choose_serializer(value: Any) -> str:
    """Pick the lightest supported serializer for a runtime value."""

    try:
        import dask.dataframe as dd  # type: ignore

        if isinstance(value, dd.DataFrame):
            return "parquet"
    except Exception:
        pass

    if _is_json_serializable(value):
        return "json"

    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.ndarray):
            return "numpy"
    except Exception:
        pass

    try:
        import torch  # type: ignore

        if (
            isinstance(value, torch.nn.Module)
            or isinstance(value, torch.Tensor)
            or isinstance(value, torch.optim.Optimizer)
        ):
            return "torch"
    except Exception:
        pass

    try:
        import pandas as pd  # type: ignore

        if isinstance(value, pd.DataFrame):
            if len(value) > 3_000_000:
                return "parquet"
            if find_spec("pyarrow") is not None:
                return "feather"
            return "pickle"
    except Exception:
        pass

    return "pickle"


def _is_json_serializable(value: Any) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def extension_for(serializer: str) -> str:
    return {
        "json": ".json",
        "numpy": ".npy",
        "pickle": ".pkl",
        "torch": ".pt",
        "feather": ".feather",
        "parquet": ".parquet",
    }.get(serializer, ".bin")


def dump_value(value: Any, serializer: str, path: Path) -> None:
    """Write a value to disk using the selected serializer."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if serializer == "json":
        with path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle)
        return
    if serializer == "numpy":
        import numpy as np  # type: ignore

        np.save(path, value, allow_pickle=False)
        return
    if serializer == "pickle":
        with path.open("wb") as handle:
            pickle.dump(value, handle)
        return
    if serializer == "torch":
        import torch  # type: ignore

        torch.save(value, path)
        return
    if serializer == "feather":
        value.to_feather(path)
        return
    if serializer == "parquet":
        try:
            import dask.dataframe as dd  # type: ignore

            if isinstance(value, dd.DataFrame):
                if value.npartitions == 1:
                    value = value.repartition(npartitions=2)
                value.to_parquet(path)
                return
        except Exception:
            pass
        value.to_parquet(path)
        return
    raise ValueError(f"Unsupported serializer: {serializer}")


def load_value(serializer: str, path: Path) -> Any:
    """Load a value from disk using the recorded serializer."""

    if serializer == "json":
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    if serializer == "numpy":
        import numpy as np  # type: ignore

        return np.load(path, allow_pickle=False)
    if serializer == "pickle":
        with path.open("rb") as handle:
            return pickle.load(handle)
    if serializer == "torch":
        import torch  # type: ignore

        return torch.load(path, weights_only=False)
    if serializer == "feather":
        import pandas as pd  # type: ignore

        return pd.read_feather(path)
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
    raise ValueError(f"Unsupported serializer: {serializer}")
