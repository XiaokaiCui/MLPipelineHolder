from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any

from ...integrations.dataframe import (
    choose_dask_serializer,
    choose_pandas_serializer,
    dump_dataframe,
    load_dataframe,
)
from ...integrations.optuna.support import is_optuna_sampler


def choose_serializer(value: Any) -> str:
    """Pick the lightest supported serializer for a runtime value."""

    if is_optuna_sampler(value):
        return "pickle"

    dask_serializer = choose_dask_serializer(value)
    if dask_serializer is not None:
        return dask_serializer

    if _is_json_safe(value):
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

    pandas_serializer = choose_pandas_serializer(value)
    if pandas_serializer is not None:
        return pandas_serializer

    return "pickle"


def _is_json_safe(value: Any) -> bool:
    """True only when a JSON round-trip preserves the exact Python value.

    ``json.dumps`` accepts values it cannot round-trip faithfully — tuples and
    sets collapse to lists, non-string dict keys are coerced to strings (so
    ``{1: "a", "1": "b"}`` loses an entry), non-finite floats emit invalid
    JSON, and integer-like scalars (e.g. ``numpy`` ints) load back as plain
    ``int``. Acceptance is therefore decided by exact-type recursion instead:
    only ``None``/``bool``/``int``/``str``/finite-``float`` leaves and
    ``list``/``dict[str, ...]`` containers qualify; everything else falls
    back to a lossless serializer.
    """
    return _json_safe_check(value, set())


def _json_safe_check(value: Any, ancestors: set[int]) -> bool:
    value_type = type(value)
    if value_type is str or value_type is bool or value is None:
        return True
    if value_type is int:
        return True
    if value_type is float:
        return math.isfinite(value)
    if value_type is list:
        value_id = id(value)
        if value_id in ancestors:
            return False
        ancestors.add(value_id)
        try:
            return all(_json_safe_check(item, ancestors) for item in value)
        finally:
            ancestors.remove(value_id)
    if value_type is dict:
        value_id = id(value)
        if value_id in ancestors:
            return False
        ancestors.add(value_id)
        try:
            return all(
                type(key) is str and _json_safe_check(item, ancestors)
                for key, item in value.items()
            )
        finally:
            ancestors.remove(value_id)
    return False


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
    if serializer in ("feather", "parquet"):
        dump_dataframe(value, serializer, path)
        return
    raise ValueError(f"Unsupported serializer: {serializer}")


def load_value(
    serializer: str,
    path: Path,
    *,
    torch_weights_only: bool = False,
) -> Any:
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

        return torch.load(path, weights_only=torch_weights_only)
    if serializer in ("feather", "parquet"):
        return load_dataframe(serializer, path)
    raise ValueError(f"Unsupported serializer: {serializer}")
