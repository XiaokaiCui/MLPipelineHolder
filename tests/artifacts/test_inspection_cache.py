from __future__ import annotations

import copyreg
import gc
import subprocess
import sys
import threading
import types
import unittest
import warnings
import weakref
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd

from mlpipelineholder import (
    InspectionCopyError,
    PipelineHandler,
    ResolutionError,
    pipeline_resolving,
)
from mlpipelineholder.execution.inspection import (
    InspectionCopier,
    InspectionMixin,
    ResolutionSource,
    _InspectionCacheBinding,
    _InspectionMemoryEstimator,
    _read_smaps_rollup_rss,
    _release_native_allocators,
)
from mlpipelineholder.persistence.artifacts.store import ArtifactStore


def return_value(value: object) -> object:
    return value


def produce_output() -> dict[str, str]:
    return {"source": "output"}


def produce_early() -> str:
    return "early"


def produce_late() -> str:
    return "late"


def append_value(value: list[int]) -> list[int]:
    value.append(99)
    return value


def aliases_shared(left: dict[str, list[int]], right: dict[str, list[int]]) -> bool:
    return left["items"] is right["items"]


class CallbackOwner:
    def __init__(self) -> None:
        self.frame = pd.DataFrame({"items": [[1, 2]]})

    def mutate(self) -> None:
        self.frame.iat[0, 0].append(3)


def invoke_callback(callback: Any) -> None:
    callback()


def invoke_append_callback(callback: Any) -> None:
    callback(2)


class WeakPayload:
    def __init__(self, size: int = 1024) -> None:
        self.data = bytearray(size)


class DeepcopyFailsButPicklable:
    def __init__(self, value: list[int]) -> None:
        self.value = value

    def __deepcopy__(self, memo: object) -> object:
        raise RuntimeError("deepcopy disabled")


class UncopyableBothWays:
    def __init__(self) -> None:
        self.lock = threading.Lock()


@dataclass
class FrameHolder:
    frame: pd.DataFrame


@dataclass(slots=True, frozen=True)
class SlottedFrameHolder:
    frame: pd.DataFrame


class PlainFrameHolder:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.self_ref: PlainFrameHolder = self


class FrameDict(dict[str, pd.DataFrame]):
    pass


class FrameList(list[pd.DataFrame]):
    pass


class TaggedTuple(tuple[Any, ...]):
    def __init__(self, iterable: Any = (), /) -> None:
        self.meta: list[int] = []


class TaggedInt(int):
    def __init__(self, value: int = 0) -> None:
        self.meta: list[int] = []


class PrivateSlotHolder:
    __slots__ = ("__payload",)

    def __init__(self) -> None:
        self.__payload = [1]

    def payload(self) -> list[int]:
        return self.__payload


class PrivateSlotChild(PrivateSlotHolder):
    __slots__ = ("__extra",)

    def __init__(self) -> None:
        super().__init__()
        self.__extra = ["child"]

    def extra(self) -> list[str]:
        return self.__extra


class SpecialList(list[Any]):
    token: str

    def __new__(cls, token: str, iterable: Any = ()) -> "SpecialList":
        obj: Any = super().__new__(cls)
        obj.token = token
        return obj

    def __init__(self, token: str, iterable: Any = ()) -> None:
        super().__init__(iterable)
        self.token = token

    def __getnewargs__(self) -> tuple[str]:
        return (self.token,)


@dataclass
class CustomState:
    value: int
    cache: list[str]

    def __getstate__(self) -> dict[str, Any]:
        return {"value": self.value, "cache": ["via-state"]}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.value = state["value"]
        self.cache = state["cache"]


class RegisteredValue:
    def __init__(self, payload: list[int]) -> None:
        self.payload = payload


class ProtocolAware:
    seen_protocol: int | None = None

    def __reduce_ex__(self, protocol: Any) -> tuple[Any, ...]:
        type(self).seen_protocol = protocol
        return (ProtocolAware, ())


class TrustedDeepcopy:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __deepcopy__(self, memo: dict[int, Any]) -> "TrustedDeepcopy":
        copied = TrustedDeepcopy("from-deepcopy")
        memo[id(self)] = copied
        return copied


class ProtocolArray(np.ndarray):
    marker: str = ""

    def __deepcopy__(self, memo: dict[int, Any] | None) -> Any:
        copied: Any = self.view(np.ndarray).copy().view(ProtocolArray)
        copied.marker = "via-protocol"
        return copied


def wrapped_frame(value: Any) -> pd.DataFrame:
    if isinstance(value, dict):
        return value["frame"]
    if isinstance(value, (list, deque)):
        return value[0]
    return value.frame


class InspectionCacheTests(unittest.TestCase):
    def _pipeline(self, root: Path) -> PipelineHandler:
        pipeline = PipelineHandler("inspection-cache", {}, root)
        pipeline.set_constant_value("first", [1, 2], to_disk=True)
        pipeline.set_constant_value("second", {"value": 3}, to_disk=True)
        return pipeline

    def test_copy_is_incremental_and_selectively_refreshes(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = self._pipeline(Path(tmp))
            with mock.patch.object(
                ArtifactStore,
                "load",
                wraps=pipeline.artifact_store.load,
            ) as load:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    pipeline.copy_for_inspection(["first"])
                self.assertEqual(load.call_count, 1)

                cached: list[int] = []
                with pipeline.inspect("first") as resolved:
                    cached = resolved.first
                    cached.append(4)
                with pipeline.inspect("first") as resolved:
                    self.assertIs(resolved.first, cached)
                    self.assertEqual(resolved.first, [1, 2, 4])

                @pipeline_resolving(pipeline)
                def investigate(first: list[int]) -> list[int]:
                    return first

                self.assertIs(investigate(), cached)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    pipeline.copy_for_inspection(["second"])
                self.assertEqual(load.call_count, 2)
                with pipeline.inspect("first") as resolved:
                    self.assertIs(resolved.first, cached)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    pipeline.copy_for_inspection(["first"])
                self.assertEqual(load.call_count, 3)
                with pipeline.inspect("first") as resolved:
                    self.assertIsNot(resolved.first, cached)
                    self.assertEqual(resolved.first, [1, 2])

    def test_load_for_inspection_alias_delegates(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = self._pipeline(Path(tmp))
            with mock.patch.object(
                pipeline,
                "copy_for_inspection",
                wraps=pipeline.copy_for_inspection,
            ) as copy:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    pipeline.load_for_inspection(["first"], compute=False)

            copy.assert_called_once()
            with pipeline.inspect("first") as resolved:
                self.assertEqual(resolved.first, [1, 2])

    def test_refresh_loaded_objects_alias_delegates(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = self._pipeline(Path(tmp))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pipeline.copy_for_inspection(["first"])

            with mock.patch.object(
                pipeline,
                "refresh_copied_objects",
                wraps=pipeline.refresh_copied_objects,
            ) as refresh:
                pipeline.refresh_loaded_objects()

            refresh.assert_called_once()

    def test_block_and_node_inspection_use_copied_objects_without_recopy(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = self._pipeline(Path(tmp))
            pipeline.set_constant_value("memory", [3, 4])
            block = pipeline.add_block("consume", 1)
            block.register_function(
                return_value,
                ["result"],
                param_mapping={"value": "first"},
            )
            memory_block = pipeline.add_block("consume_memory", 2)
            memory_block.register_function(
                return_value,
                ["result2"],
                param_mapping={"value": "memory"},
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pipeline.copy_for_inspection(["first", "memory"])
            disk_cached = pipeline._inspection_cache_entries[
                pipeline._inspection_cache_bindings["first"].cache_key
            ].value
            memory_cached = pipeline._inspection_cache_entries[
                pipeline._inspection_cache_bindings["memory"].cache_key
            ].value

            with (
                mock.patch.object(
                    ArtifactStore,
                    "load",
                    wraps=pipeline.artifact_store.load,
                ) as load,
                warnings.catch_warnings(),
            ):
                warnings.simplefilter("ignore", UserWarning)
                protected = block.inspect(
                    resolve_only=True,
                    allow_mutable_objects=False,
                )
                shared = block.inspect(resolve_only=True)
                node = pipeline.inspect_node(
                    node_name="consume",
                    resolve_only=True,
                )
                memory_protected = memory_block.inspect(
                    resolve_only=True,
                    allow_mutable_objects=False,
                )
                self.assertEqual(load.call_count, 0)

            self.assertIs(protected.arguments["value"], disk_cached)
            self.assertIs(shared.arguments["value"], disk_cached)
            self.assertIs(node.arguments["value"], disk_cached)
            self.assertIs(memory_protected.arguments["value"], memory_cached)

    def test_aliased_names_load_and_copy_shared_targets_once(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("aliases", {}, Path(tmp) / "project")
            record = pipeline.artifact_store.save(
                variable_name="shared",
                value={"payload": [1, 2]},
                block_name="producer",
                function_name="produce",
                run_id="run",
            )
            pipeline.set_constant_value("shared_artifact_a", record, copy=False)
            pipeline.set_constant_value("shared_artifact_b", record, copy=False)
            shared_object = [3, 4]
            pipeline.set_constant_value("shared_memory_a", shared_object, copy=False)
            pipeline.set_constant_value("shared_memory_b", shared_object, copy=False)

            with (
                mock.patch.object(
                    ArtifactStore,
                    "load",
                    wraps=pipeline.artifact_store.load,
                ) as load,
                mock.patch.object(
                    pipeline,
                    "_isolate_inspection_value",
                    wraps=pipeline._isolate_inspection_value,
                ) as isolate,
                warnings.catch_warnings(),
            ):
                warnings.simplefilter("ignore", UserWarning)
                pipeline.copy_for_inspection(
                    [
                        "shared_artifact_a",
                        "shared_artifact_b",
                        "shared_memory_a",
                        "shared_memory_b",
                    ]
                )
            self.assertEqual(load.call_count, 1)
            self.assertEqual(isolate.call_count, 1)

            with pipeline.inspect("shared_artifact_a", "shared_artifact_b") as resolved:
                self.assertIs(resolved.shared_artifact_a, resolved.shared_artifact_b)
            with pipeline.inspect("shared_memory_a", "shared_memory_b") as resolved:
                self.assertIs(resolved.shared_memory_a, resolved.shared_memory_b)

    def test_batch_copy_preserves_nested_aliases(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("batch-aliases", {}, Path(tmp) / "project")
            shared = [1, 2]
            pipeline.set_constant_value("left", {"items": shared}, copy=False)
            pipeline.set_constant_value("right", {"items": shared}, copy=False)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pipeline.copy_for_inspection(["left", "right"])

            cached_items: list[int] = []
            with pipeline.inspect("left", "right") as resolved:
                self.assertIs(
                    resolved.left["items"],
                    resolved.right["items"],
                )
                cached_items = resolved.left["items"]
            self.assertIsNot(
                pipeline.get_constant_value("left")["items"],
                cached_items,
            )

    def test_block_inspection_alias_behaviour_matches_uncached_after_copy(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("block-aliases", {}, Path(tmp) / "project")
            shared = [1, 2]
            pipeline.set_constant_value("left", {"items": shared}, copy=False)
            pipeline.set_constant_value("right", {"items": shared}, copy=False)
            block = pipeline.add_block("check", 1)
            block.register_function(aliases_shared, ["result"])

            uncached = block.inspect(
                resolve_only=True,
                allow_mutable_objects=False,
            )
            self.assertIs(
                uncached.arguments["left"]["items"],
                uncached.arguments["right"]["items"],
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pipeline.copy_for_inspection(["left", "right"])

            cached = block.inspect(
                resolve_only=True,
                allow_mutable_objects=False,
            )
            self.assertTrue(
                cached.arguments["left"]["items"]
                is cached.arguments["right"]["items"]
            )
            self.assertTrue(block.inspect(allow_mutable_objects=False))

    def test_identity_guard_outranks_reused_id(self) -> None:
        class Refable:
            pass

        guard = Refable()
        replacement = Refable()
        binding = _InspectionCacheBinding(
            cache_key=(("memory", id(replacement)), True),
            kind="memory",
            priority=None,
            compute=True,
            original_ref=None,
            original_guard=guard,
            original_id=id(replacement),
        )

        self.assertFalse(
            InspectionMixin._inspection_original_matches(binding, replacement)
        )

    def test_replaced_non_weakrefable_value_is_not_served_stale(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = self._pipeline(Path(tmp))
            pipeline.set_constant_value("items", [1], copy=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pipeline.copy_for_inspection(["items"])

            binding = pipeline._inspection_cache_bindings["items"]
            self.assertIsNotNone(binding.original_guard)

            pipeline.set_constant_value("items", [2, 3], copy=False)
            with pipeline.inspect("items") as resolved:
                self.assertEqual(resolved.items, [2, 3])
            self.assertIsNone(binding.original_guard)

    def test_refresh_and_unload_apply_to_all_copied_objects(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = self._pipeline(Path(tmp))
            pipeline.set_constant_value("memory", [5, 6])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pipeline.copy_for_inspection(["first", "second", "memory"])

            with pipeline.inspect("first", "second", "memory") as resolved:
                resolved.first.append(8)
                resolved.second["value"] = 9
                resolved.memory.append(10)
            pipeline.refresh_copied_objects()
            with pipeline.inspect("first", "second", "memory") as resolved:
                self.assertEqual(resolved.first, [1, 2])
                self.assertEqual(resolved.second, {"value": 3})
                self.assertEqual(resolved.memory, [5, 6])
            self.assertEqual(pipeline.get_constant_value("memory"), [5, 6])

            pipeline.unload_inspection_objects()
            self.assertEqual(pipeline._inspection_cache_bindings, {})
            with pipeline.inspect("first") as resolved:
                self.assertEqual(resolved.first, [1, 2])
            with self.assertWarnsRegex(UserWarning, "nothing was unloaded"):
                pipeline.unload_inspection_objects()
            with self.assertWarnsRegex(UserWarning, "nothing was refreshed"):
                pipeline.refresh_copied_objects()

    def test_invalid_names_and_unknown_targets_are_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = self._pipeline(Path(tmp))

            with self.assertRaisesRegex(ResolutionError, "missing"):
                pipeline.copy_for_inspection(["first", "missing"])
            self.assertEqual(pipeline._inspection_cache_bindings, {})

            with self.assertRaisesRegex(ValueError, "must be unique"):
                pipeline.copy_for_inspection(["first", "first"])
            with self.assertRaisesRegex(ValueError, "non-empty"):
                pipeline.copy_for_inspection([])
            with self.assertRaisesRegex(TypeError, "list or tuple"):
                pipeline.copy_for_inspection(
                    "first"  # pyright: ignore[reportArgumentType]
                )
            with self.assertRaisesRegex(TypeError, "integer priority group"):
                pipeline.copy_for_inspection(
                    ["first"],
                    priority=1.5,  # pyright: ignore[reportArgumentType]
                )

    def test_in_memory_values_are_copied_and_isolated(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = self._pipeline(Path(tmp))
            pipeline.set_constant_value("memory", [7, 8])
            pipeline.set_config("threshold", 3)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pipeline.copy_for_inspection(["memory", "threshold"])

            cached: list[int] = []
            with pipeline.inspect("memory", "threshold") as resolved:
                resolved.memory.append(9)
                cached = resolved.memory
                self.assertEqual(resolved.threshold, 3)

            self.assertEqual(pipeline.get_constant_value("memory"), [7, 8])
            with pipeline.inspect("memory") as resolved:
                self.assertIs(resolved.memory, cached)
                self.assertEqual(resolved.memory, [7, 8, 9])

    def test_pandas_object_cells_are_isolated(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler(
                "pandas-isolation",
                {},
                Path(tmp) / "project",
            )
            frame = pd.DataFrame(
                {
                    "nested": [[1, 2]],
                    "meta": pd.Series([{"flags": ["selected"]}], dtype=object),
                }
            )
            series = pd.Series([[7, 8]], dtype=object)
            pipeline.set_constant_value("frame", frame, copy=False)
            pipeline.set_constant_value("series", series, copy=False)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pipeline.copy_for_inspection(["frame", "series"])

            with pipeline.inspect("frame", "series") as resolved:
                resolved.frame.iloc[0, 0].append(3)
                resolved.frame.iloc[0, 1]["flags"].append("experimental")
                resolved.series.iloc[0].append(10)

            self.assertEqual(frame.iloc[0, 0], [1, 2])
            self.assertEqual(frame.iloc[0, 1], {"flags": ["selected"]})
            self.assertEqual(series.iloc[0], [7, 8])

            with pipeline.inspect("frame", "series") as resolved:
                self.assertEqual(resolved.frame.iloc[0, 0], [1, 2, 3])
                self.assertEqual(
                    resolved.frame.iloc[0, 1],
                    {"flags": ["selected", "experimental"]},
                )
                self.assertEqual(resolved.series.iloc[0], [7, 8, 10])

    def test_nested_pandas_frame_is_isolated_in_cached_and_protected_inspection(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("nested-frame", {}, Path(tmp))
            original = pd.DataFrame({"items": [[1, 2]]})
            wrapped = {"frame": original}
            pipeline.set_constant_value("wrapped", wrapped, copy=False)
            block = pipeline.add_block("use", 1)
            block.register_function(
                return_value, ["result"], param_mapping={"value": "wrapped"}
            )

            protected = block.inspect(resolve_only=True, allow_mutable_objects=False)
            protected.arguments["value"]["frame"].iat[0, 0].append(3)
            self.assertEqual(original.iat[0, 0], [1, 2])

            pipeline.copy_for_inspection(["wrapped"])
            with pipeline.inspect("wrapped") as resolved:
                resolved.wrapped["frame"].iat[0, 0].append(4)
            self.assertEqual(original.iat[0, 0], [1, 2])

    def test_pandas_inside_object_array_is_isolated(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("object-array-frame", {}, Path(tmp))
            frame = pd.DataFrame({"items": [[1, 2]]})
            original = np.empty(1, dtype=object)
            original[0] = frame
            pipeline.set_constant_value("array", original, copy=False)
            pipeline.set_constant_value("frame", frame, copy=False)
            pipeline.copy_for_inspection(["array", "frame"])

            with pipeline.inspect("array", "frame") as resolved:
                self.assertIs(resolved.array[0], resolved.frame)
                resolved.array[0].iat[0, 0].append(3)
            self.assertEqual(frame.iat[0, 0], [1, 2])

    def test_wrapped_pandas_frame_is_isolated_in_cache_and_protected_inspection(self) -> None:
        wrappers = (
            lambda frame: FrameHolder(frame),
            lambda frame: SlottedFrameHolder(frame),
            lambda frame: PlainFrameHolder(frame),
            lambda frame: OrderedDict(frame=frame),
            lambda frame: defaultdict(list, frame=frame),
            lambda frame: FrameDict(frame=frame),
            lambda frame: FrameList([frame]),
            lambda frame: types.SimpleNamespace(frame=frame),
            lambda frame: deque([frame]),
        )
        for wrap in wrappers:
            with self.subTest(wrapper=wrap):
                with TemporaryDirectory() as tmp:
                    pipeline = PipelineHandler("wrapped-frame", {}, Path(tmp))
                    frame = pd.DataFrame({"items": [[1, 2]]})
                    wrapped = wrap(frame)
                    pipeline.set_constant_value("wrapped", wrapped, copy=False)
                    block = pipeline.add_block("use", 1)
                    block.register_function(
                        return_value,
                        ["result"],
                        param_mapping={"value": "wrapped"},
                    )

                    protected = block.inspect(
                        resolve_only=True,
                        allow_mutable_objects=False,
                    ).arguments["value"]
                    protected_frame = wrapped_frame(protected)
                    self.assertIsNot(protected_frame, frame)
                    protected_frame.iat[0, 0].append(3)
                    self.assertEqual(frame.iat[0, 0], [1, 2])

                    pipeline.copy_for_inspection(["wrapped"])
                    with pipeline.inspect("wrapped") as resolved:
                        cached_frame = wrapped_frame(resolved.wrapped)
                        self.assertIsNot(cached_frame, frame)
                        cached_frame.iat[0, 0].append(4)
                    self.assertEqual(frame.iat[0, 0], [1, 2])

    def test_self_referential_object_preserves_cycle(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("self-ref", {}, Path(tmp))
            original = PlainFrameHolder(pd.DataFrame({"items": [[1, 2]]}))
            original.self_ref = original
            pipeline.set_constant_value("holder", original, copy=False)
            block = pipeline.add_block("use", 1)
            block.register_function(
                return_value,
                ["result"],
                param_mapping={"value": "holder"},
            )

            protected = block.inspect(
                resolve_only=True,
                allow_mutable_objects=False,
            ).arguments["value"]
            self.assertIsNot(protected, original)
            self.assertIs(protected.self_ref, protected)

            pipeline.copy_for_inspection(["holder"])
            with pipeline.inspect("holder") as resolved:
                copied = resolved.holder
                self.assertIsNot(copied, original)
                self.assertIs(copied.self_ref, copied)
                copied.frame.iat[0, 0].append(3)
            self.assertEqual(original.frame.iat[0, 0], [1, 2])

    def test_wrapped_batch_copies_keep_shared_frame_alias(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("wrapped-alias", {}, Path(tmp))
            frame = pd.DataFrame({"items": [[1, 2]]})
            pipeline.set_constant_value("left", FrameHolder(frame), copy=False)
            pipeline.set_constant_value("right", {"frame": frame}, copy=False)
            pipeline.copy_for_inspection(["left", "right"])

            with pipeline.inspect("left", "right") as resolved:
                self.assertIs(resolved.left.frame, resolved.right["frame"])
                resolved.left.frame.iat[0, 0].append(3)
            self.assertEqual(frame.iat[0, 0], [1, 2])

    def test_bound_method_copy_isolates_owner(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("bound-method", {}, Path(tmp))
            owner = CallbackOwner()
            pipeline.set_constant_value("callback", owner.mutate, copy=False)
            block = pipeline.add_block("use", 1)
            block.register_function(
                invoke_callback,
                ["result"],
                param_mapping={"callback": "callback"},
            )

            resolved = block.inspect(
                resolve_only=True,
                allow_mutable_objects=False,
            )
            copied_callback = resolved.arguments["callback"]
            self.assertIsNot(copied_callback, owner.mutate)
            self.assertIsNot(copied_callback.__self__, owner)

            block.inspect(allow_mutable_objects=False)
            self.assertEqual(owner.frame.iat[0, 0], [1, 2])

            pipeline.copy_for_inspection(["callback"])
            block.inspect(allow_mutable_objects=False)
            self.assertEqual(owner.frame.iat[0, 0], [1, 2])

    def test_bound_builtin_method_copy_isolates_owner(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("bound-builtin", {}, Path(tmp))
            items = [1]
            pipeline.set_constant_value("callback", items.append, copy=False)
            block = pipeline.add_block("use", 1)
            block.register_function(
                invoke_append_callback,
                ["result"],
                param_mapping={"callback": "callback"},
            )

            resolved = block.inspect(
                resolve_only=True,
                allow_mutable_objects=False,
            )
            copied_callback = resolved.arguments["callback"]
            self.assertIsNot(copied_callback, items.append)
            self.assertIsNot(copied_callback.__self__, items)

            block.inspect(allow_mutable_objects=False)
            self.assertEqual(items, [1])

            pipeline.copy_for_inspection(["callback"])
            block.inspect(allow_mutable_objects=False)
            self.assertEqual(items, [1])

    def test_immutable_subclass_state_is_isolated(self) -> None:
        factories = (
            lambda: TaggedTuple((1, 2)),
            lambda: TaggedInt(7),
        )
        for factory in factories:
            with self.subTest(factory=factory):
                with TemporaryDirectory() as tmp:
                    pipeline = PipelineHandler("tagged", {}, Path(tmp))
                    value = factory()
                    value.meta = [3]
                    pipeline.set_constant_value("value", value, copy=False)
                    block = pipeline.add_block("use", 1)
                    block.register_function(
                        return_value,
                        ["result"],
                        param_mapping={"value": "value"},
                    )

                    protected = block.inspect(
                        resolve_only=True,
                        allow_mutable_objects=False,
                    ).arguments["value"]
                    self.assertIsNot(protected, value)
                    self.assertEqual(protected.meta, [3])
                    protected.meta.append(4)
                    self.assertEqual(value.meta, [3])

                    pipeline.copy_for_inspection(["value"])
                    with pipeline.inspect("value") as resolved:
                        cached = resolved.value
                        self.assertIsNot(cached, value)
                        self.assertEqual(cached.meta, [3])
                        cached.meta.append(5)
                    self.assertEqual(value.meta, [3])

    def test_private_slot_state_is_copied(self) -> None:
        for holder_type in (PrivateSlotHolder, PrivateSlotChild):
            with self.subTest(holder=holder_type):
                with TemporaryDirectory() as tmp:
                    pipeline = PipelineHandler("private-slots", {}, Path(tmp))
                    holder = holder_type()
                    pipeline.set_constant_value("holder", holder, copy=False)
                    block = pipeline.add_block("use", 1)
                    block.register_function(
                        return_value,
                        ["result"],
                        param_mapping={"value": "holder"},
                    )

                    protected = block.inspect(
                        resolve_only=True,
                        allow_mutable_objects=False,
                    ).arguments["value"]
                    self.assertEqual(protected.payload(), [1])
                    if isinstance(protected, PrivateSlotChild):
                        self.assertEqual(protected.extra(), ["child"])
                    protected.payload().append(3)
                    self.assertEqual(holder.payload(), [1])

                    pipeline.copy_for_inspection(["holder"])
                    with pipeline.inspect("holder") as resolved:
                        cached = resolved.holder
                        self.assertIsNot(cached, holder)
                        self.assertEqual(cached.payload(), [1])
                        if isinstance(cached, PrivateSlotChild):
                            self.assertEqual(cached.extra(), ["child"])
                        cached.payload().append(4)
                    self.assertEqual(holder.payload(), [1])

    def test_container_subclass_with_custom_new_round_trips(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("special-list", {}, Path(tmp))
            frame = pd.DataFrame({"items": [[1, 2]]})
            value = SpecialList("tok", [frame])
            pipeline.set_constant_value("value", value, copy=False)
            block = pipeline.add_block("use", 1)
            block.register_function(
                return_value,
                ["result"],
                param_mapping={"value": "value"},
            )

            protected = block.inspect(
                resolve_only=True,
                allow_mutable_objects=False,
            ).arguments["value"]
            self.assertIsInstance(protected, SpecialList)
            self.assertEqual(protected.token, "tok")
            self.assertEqual(len(protected), 1)
            self.assertIsNot(protected[0], frame)
            protected[0].iat[0, 0].append(3)
            self.assertEqual(frame.iat[0, 0], [1, 2])

            pipeline.copy_for_inspection(["value"])
            with pipeline.inspect("value") as resolved:
                cached = resolved.value
                self.assertIsInstance(cached, SpecialList)
                self.assertEqual(cached.token, "tok")
                self.assertEqual(len(cached), 1)
                self.assertIsNot(cached[0], frame)
                cached[0].iat[0, 0].append(4)
            self.assertEqual(frame.iat[0, 0], [1, 2])

    def test_dataclass_custom_state_hooks_are_honored(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("custom-state", {}, Path(tmp))
            original = CustomState(1, ["original"])
            pipeline.set_constant_value("state", original, copy=False)
            block = pipeline.add_block("use", 1)
            block.register_function(
                return_value,
                ["result"],
                param_mapping={"value": "state"},
            )

            protected = block.inspect(
                resolve_only=True,
                allow_mutable_objects=False,
            ).arguments["value"]
            self.assertEqual(protected.cache, ["via-state"])

            pipeline.copy_for_inspection(["state"])
            with pipeline.inspect("state") as resolved:
                self.assertEqual(resolved.state.cache, ["via-state"])
            self.assertEqual(original.cache, ["original"])

    def test_copyreg_registered_reducer_is_honoured(self) -> None:
        calls: list[list[int]] = []

        def reducer(value: RegisteredValue) -> tuple[Any, ...]:
            calls.append(value.payload)
            return (RegisteredValue, (value.payload,))

        copyreg.pickle(RegisteredValue, reducer)
        try:
            with TemporaryDirectory() as tmp:
                pipeline = PipelineHandler("copyreg", {}, Path(tmp))
                original = RegisteredValue([1, 2])
                pipeline.set_constant_value("value", original, copy=False)
                block = pipeline.add_block("use", 1)
                block.register_function(
                    return_value,
                    ["result"],
                    param_mapping={"value": "value"},
                )

                protected = block.inspect(
                    resolve_only=True,
                    allow_mutable_objects=False,
                ).arguments["value"]
                self.assertIsNot(protected, original)
                self.assertEqual(protected.payload, [1, 2])

                pipeline.copy_for_inspection(["value"])
                with pipeline.inspect("value") as resolved:
                    cached = resolved.value
                    self.assertIsNot(cached, original)
                    self.assertEqual(cached.payload, [1, 2])
        finally:
            copyreg.dispatch_table.pop(RegisteredValue, None)
        self.assertEqual(len(calls), 2)

    def test_reduce_ex_receives_deepcopy_protocol(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("protocol", {}, Path(tmp))
            original = ProtocolAware()
            pipeline.set_constant_value("value", original, copy=False)
            block = pipeline.add_block("use", 1)
            block.register_function(
                return_value,
                ["result"],
                param_mapping={"value": "value"},
            )

            ProtocolAware.seen_protocol = None
            block.inspect(resolve_only=True, allow_mutable_objects=False)
            self.assertEqual(ProtocolAware.seen_protocol, 4)

            ProtocolAware.seen_protocol = None
            pipeline.copy_for_inspection(["value"])
            with pipeline.inspect("value") as resolved:
                self.assertIsInstance(resolved.value, ProtocolAware)
            self.assertEqual(ProtocolAware.seen_protocol, 4)

    def test_explicit_deepcopy_is_trusted(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("trusted-deepcopy", {}, Path(tmp))
            original = TrustedDeepcopy("original")
            pipeline.set_constant_value("value", original, copy=False)
            block = pipeline.add_block("use", 1)
            block.register_function(
                return_value,
                ["result"],
                param_mapping={"value": "value"},
            )

            protected = block.inspect(
                resolve_only=True,
                allow_mutable_objects=False,
            ).arguments["value"]
            self.assertEqual(protected.marker, "from-deepcopy")

            pipeline.copy_for_inspection(["value"])
            with pipeline.inspect("value") as resolved:
                self.assertEqual(resolved.value.marker, "from-deepcopy")

    def test_exact_object_ndarray_subclass_follows_its_protocol(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("protocol-array", {}, Path(tmp))
            original = np.empty(1, dtype=object).view(ProtocolArray)
            pipeline.set_constant_value("array", original, copy=False)
            block = pipeline.add_block("use", 1)
            block.register_function(
                return_value,
                ["result"],
                param_mapping={"value": "array"},
            )

            protected = block.inspect(
                resolve_only=True,
                allow_mutable_objects=False,
            ).arguments["value"]
            self.assertIsInstance(protected, ProtocolArray)
            self.assertEqual(protected.marker, "via-protocol")

    def test_structured_object_ndarray_is_isolated(self) -> None:
        dtype = np.dtype([("frame", object), ("score", np.int64)])
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("structured-array", {}, Path(tmp))
            frame = pd.DataFrame({"items": [[1, 2]]})
            original = np.empty(1, dtype=dtype)
            original["frame"][0] = frame
            original["score"][0] = 10
            pipeline.set_constant_value("array", original, copy=False)
            block = pipeline.add_block("use", 1)
            block.register_function(
                return_value,
                ["result"],
                param_mapping={"value": "array"},
            )

            protected = block.inspect(
                resolve_only=True,
                allow_mutable_objects=False,
            ).arguments["value"]
            self.assertEqual(protected["score"][0], 10)
            self.assertIsNot(protected["frame"][0], frame)
            protected["frame"][0].iat[0, 0].append(3)
            self.assertEqual(frame.iat[0, 0], [1, 2])

            pipeline.copy_for_inspection(["array"])
            with pipeline.inspect("array") as resolved:
                cached = resolved.array
                self.assertEqual(cached["score"][0], 10)
                self.assertIsNot(cached["frame"][0], frame)
                cached["frame"][0].iat[0, 0].append(4)
            self.assertEqual(frame.iat[0, 0], [1, 2])

    def test_nested_structured_object_ndarray_is_isolated(self) -> None:
        inner = np.dtype([("frame", object), ("label", object)])
        dtype = np.dtype([("inner", inner), ("score", np.int64)])
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("nested-structured-array", {}, Path(tmp))
            frame = pd.DataFrame({"items": [[1, 2]]})
            original = np.empty(1, dtype=dtype)
            original["inner"]["frame"][0] = frame
            original["inner"]["label"][0] = "kept"
            original["score"][0] = 10
            pipeline.set_constant_value("array", original, copy=False)
            pipeline.copy_for_inspection(["array"])

            with pipeline.inspect("array") as resolved:
                cached = resolved.array
                self.assertEqual(cached["inner"]["label"][0], "kept")
                self.assertEqual(cached["score"][0], 10)
                self.assertIsNot(cached["inner"]["frame"][0], frame)
                cached["inner"]["frame"][0].iat[0, 0].append(3)
            self.assertEqual(frame.iat[0, 0], [1, 2])

    def test_copier_keeps_originals_alive_for_id_safety(self) -> None:
        copier = InspectionCopier(None, allow_mutable_objects=False)
        original = WeakPayload()
        reference = weakref.ref(original)
        copied, _ = copier.prepare(
            original,
            parameter_name="value",
            source=ResolutionSource(kind="test"),
        )
        self.assertIsNot(copied, original)
        del original
        gc.collect()
        self.assertIsNotNone(reference())
        self.assertTrue(any(item is reference() for item in copier._keepalive))

    def test_callable_identity_rules_agree_with_prepare(self) -> None:
        copier = InspectionCopier(None, allow_mutable_objects=False)
        source = ResolutionSource(kind="test")
        for template in (len, str.maketrans, int):
            with self.subTest(template=template):
                self.assertFalse(copier.requires_copy(template))
                copied, _ = copier.prepare(
                    template,
                    parameter_name="value",
                    source=source,
                )
                self.assertIs(copied, template)
        self.assertTrue(copier.requires_copy([1].append))
        self.assertTrue(copier.requires_copy([1].__iter__))

    def test_copier_does_not_import_optional_libraries(self) -> None:
        script = (
            "import sys\n"
            "from mlpipelineholder.execution.inspection import (\n"
            "    InspectionCopier,\n"
            "    ResolutionSource,\n"
            ")\n"
            "OPTIONAL = ('pandas', 'numpy', 'dask', 'torch', 'optuna', 'pyarrow')\n"
            "before = {name for name in OPTIONAL if name in sys.modules}\n"
            "class Value:\n"
            "    def __init__(self):\n"
            "        self.items = [1, 2]\n"
            "value = Value()\n"
            "copied, _ = InspectionCopier(None, allow_mutable_objects=False).prepare(\n"
            "    value,\n"
            "    parameter_name='value',\n"
            "    source=ResolutionSource(kind='test'),\n"
            ")\n"
            "assert copied is not value\n"
            "assert copied.items == [1, 2]\n"
            "after = {name for name in OPTIONAL if name in sys.modules}\n"
            "print(','.join(sorted(after - before)))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(result.stdout.strip(), "")

    @unittest.skipUnless(find_spec("torch") is not None, "torch is not installed")
    def test_torch_module_and_optimizer_keep_deepcopy_boundary(self) -> None:
        import torch

        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("torch-boundary", {}, Path(tmp))
            module = torch.nn.Linear(2, 2)
            optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
            pipeline.set_constant_value("module", module, copy=False)
            pipeline.set_constant_value("optimizer", optimizer, copy=False)
            block = pipeline.add_block("use", 1)
            block.register_function(
                return_value,
                ["result"],
                param_mapping={"value": "module"},
            )

            before = module.weight.detach().clone()
            protected = block.inspect(
                resolve_only=True,
                allow_mutable_objects=False,
            ).arguments["value"]
            self.assertIsInstance(next(protected.parameters()), torch.nn.Parameter)

            pipeline.copy_for_inspection(["module", "optimizer"])
            with pipeline.inspect("module", "optimizer") as resolved:
                copied_module = resolved.module
                copied_optimizer = resolved.optimizer
                self.assertIsNot(copied_module, module)
                self.assertIsNot(copied_optimizer, optimizer)
                self.assertIsInstance(
                    next(copied_module.parameters()),
                    torch.nn.Parameter,
                )
                self.assertIsInstance(
                    copied_optimizer.param_groups[0]["params"][0],
                    torch.nn.Parameter,
                )
                with torch.no_grad():
                    copied_module.weight.add_(1)
            self.assertTrue(torch.equal(module.weight, before))

    def test_failed_copy_does_not_expose_partial_pandas_memo_to_later_names(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("partial-memo", {}, Path(tmp))
            cell = DeepcopyFailsButPicklable([1])
            frame = pd.DataFrame({"nested": [cell]})
            pipeline.set_constant_value("frame", frame, copy=False)
            pipeline.set_constant_value("wrapped", {"frame": frame}, copy=False)

            with mock.patch.object(pipeline.logger, "warning") as warning:
                pipeline.copy_for_inspection(["frame", "wrapped"])
            self.assertNotIn("frame", pipeline._inspection_cache_bindings)
            self.assertTrue(warning.called)
            with pipeline.inspect("wrapped") as resolved:
                copied_cell = resolved.wrapped["frame"].iat[0, 0]
                self.assertIsNot(copied_cell, cell)
                copied_cell.value.append(2)
            self.assertEqual(cell.value, [1])

    def test_cyclic_standard_containers_still_preserve_aliases(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("recursive-copy", {}, Path(tmp))
            original: list[object] = []
            recursive_tuple = (original,)
            original.append(recursive_tuple)
            pipeline.set_constant_value("recursive", original, copy=False)
            pipeline.copy_for_inspection(["recursive"])
            with pipeline.inspect("recursive") as resolved:
                copied = resolved.recursive
                self.assertIsNot(copied, original)
                self.assertIs(copied[0][0], copied)

    def test_dirty_storage_value_is_copied_instead_of_stale_artifact(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler(
                "dirty-storage",
                {},
                Path(tmp) / "project",
            )
            hash_id = pipeline.save_to_storage("model_data", {"version": 1})
            pipeline.update_storage(hash_id, {"version": 1}, to_disk=True)
            pipeline.update_storage(hash_id, {"version": 2}, to_disk=False)

            uncached: object = None
            with pipeline.inspect("model_data") as resolved:
                uncached = resolved.model_data
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pipeline.copy_for_inspection(["model_data"])
            with pipeline.inspect("model_data") as resolved:
                self.assertEqual(resolved.model_data, uncached)
            self.assertEqual(uncached, {"version": 2})

    def test_dead_weakref_binding_does_not_match_by_id(self) -> None:
        class Refable:
            pass

        original = Refable()
        binding = _InspectionCacheBinding(
            cache_key=(("memory", id(original)), True),
            kind="memory",
            priority=None,
            compute=True,
            original_ref=weakref.ref(original),
            original_id=id(original),
        )
        del original
        gc.collect()

        self.assertFalse(
            InspectionMixin._inspection_original_matches(binding, Refable())
        )

    def test_replacing_in_memory_original_with_artifact_releases_identity_guard(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("guard-source-change", {}, Path(tmp))
            pipeline.set_constant_value("items", [1], copy=False)
            pipeline.copy_for_inspection(["items"])
            binding = pipeline._inspection_cache_bindings["items"]
            self.assertIsNotNone(binding.original_guard)

            pipeline.set_constant_value("items", [2], to_disk=True)
            with pipeline.inspect("items") as resolved:
                self.assertEqual(resolved.items, [2])
            self.assertIsNone(binding.original_guard)

    def test_skipped_nudge_rearms_after_original_dies(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = self._pipeline(Path(tmp))
            uncopyable = UncopyableBothWays()
            pipeline.set_constant_value("lock", uncopyable)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pipeline.copy_for_inspection(["lock"])
            self.assertIn("lock", pipeline._inspection_copy_skipped)

            del uncopyable
            pipeline.set_constant_value("lock", [1, 2])
            gc.collect()

            with mock.patch.object(pipeline.logger, "info") as info:
                with pipeline.inspect("lock") as resolved:
                    self.assertEqual(resolved.lock, [1, 2])

            messages = [call.args[0] for call in info.call_args_list]
            self.assertTrue(
                any(
                    "copy it first with copy_for_inspection" in message
                    for message in messages
                )
            )

    def test_deepcopy_failure_falls_back_to_temporary_serialization(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline = PipelineHandler("fallback", {}, root / "project")
            original = DeepcopyFailsButPicklable([1, 2])
            pipeline.set_constant_value("payload", original)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pipeline.copy_for_inspection(["payload"])

            with pipeline.inspect("payload") as resolved:
                self.assertIsNot(resolved.payload, original)
                self.assertEqual(resolved.payload.value, [1, 2])
            self.assertEqual(original.value, [1, 2])
            temp_root = root / "project" / "inspection_tmp"
            self.assertEqual(list(temp_root.glob("*")), [])

    def test_uncopyable_object_is_skipped_with_save_reload_guidance(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = self._pipeline(Path(tmp))
            original = UncopyableBothWays()
            pipeline.set_constant_value("lock", original)

            with (
                mock.patch.object(pipeline.logger, "warning") as warning,
                warnings.catch_warnings(),
            ):
                warnings.simplefilter("ignore", UserWarning)
                pipeline.copy_for_inspection(["first", "lock"])

            messages = [call.args[0] for call in warning.call_args_list]
            self.assertTrue(
                any("save the pipeline before inspection" in message for message in messages)
            )
            self.assertIn("first", pipeline._inspection_cache_bindings)
            self.assertNotIn("lock", pipeline._inspection_cache_bindings)

            with pipeline.inspect("lock") as resolved:
                self.assertIs(resolved.lock, original)

    def test_refresh_failure_raises_and_keeps_cache(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = self._pipeline(Path(tmp))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pipeline.copy_for_inspection(["first"])
            before: list[int] = []
            with pipeline.inspect("first") as resolved:
                before = resolved.first

            binding_before = pipeline._inspection_cache_bindings["first"]
            pipeline.set_constant_value("first", UncopyableBothWays())
            with self.assertRaisesRegex(InspectionCopyError, "Could not refresh"):
                pipeline.refresh_copied_objects()

            self.assertIs(
                pipeline._inspection_cache_bindings["first"],
                binding_before,
            )
            self.assertIs(
                pipeline._inspection_cache_entries[binding_before.cache_key].value,
                before,
            )
            self.assertIsInstance(
                pipeline.get_constant_value("first"),
                UncopyableBothWays,
            )

    def test_priority_copy_is_used_by_priority_matched_inspection(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("priority", {}, Path(tmp) / "project")
            early = pipeline.add_block("early", 1)
            early.register_function(
                produce_early,
                ["generation"],
                save_to_disk=["generation"],
            )
            late = pipeline.add_block("late", 2)
            late.register_function(
                produce_late,
                ["generation"],
                save_to_disk=["generation"],
            )
            pipeline.run_all()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pipeline.copy_for_inspection(["generation"], priority=2)

            with pipeline.inspect("generation", priority=2) as resolved:
                self.assertEqual(resolved.generation, "early")
            with pipeline.inspect("generation") as resolved:
                self.assertEqual(resolved.generation, "late")

    def test_priority_views_match_uncached_resolution(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("priority-views", {}, Path(tmp) / "project")
            pipeline.add_block("early", 1).register_function(
                produce_early,
                ["generation"],
                save_to_disk=["generation"],
            )
            pipeline.add_block("late", 2).register_function(
                produce_late,
                ["generation"],
                save_to_disk=["generation"],
            )
            pipeline.run_all()

            # The selected node's own output is excluded from its view, so both
            # priority=2 and priority=3 inspect the state before the late node.
            uncached_priority_view = ""
            with pipeline.inspect("generation", priority=3) as resolved:
                uncached_priority_view = resolved.generation
            self.assertEqual(uncached_priority_view, "early")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pipeline.copy_for_inspection(["generation"], priority=2)

            with mock.patch.object(
                ArtifactStore,
                "load",
                wraps=pipeline.artifact_store.load,
            ) as load:
                with pipeline.inspect("generation", priority=3) as resolved:
                    self.assertEqual(resolved.generation, uncached_priority_view)
                self.assertEqual(load.call_count, 0)

                with pipeline.inspect("generation") as resolved:
                    self.assertEqual(resolved.generation, "late")
                self.assertEqual(load.call_count, 1)

    def test_priority_mismatch_serves_cache_with_reminder(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("reminder", {}, Path(tmp) / "project")
            pipeline.add_block("producer", 1).register_function(
                produce_early,
                ["value"],
            )
            pipeline.set_constant_value("disk", [1, 2], to_disk=True)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pipeline.copy_for_inspection(["disk"], priority=1)

            with (
                mock.patch.object(pipeline.logger, "info") as info,
                warnings.catch_warnings(),
            ):
                warnings.simplefilter("ignore", UserWarning)
                with pipeline.inspect("disk", priority=2) as resolved:
                    self.assertEqual(resolved.disk, [1, 2])

            messages = [call.args[0] for call in info.call_args_list]
            self.assertTrue(
                any("Requested priority=2" in message for message in messages)
            )

    def test_miss_nudges_recommend_copy_for_inspection(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = self._pipeline(Path(tmp))
            pipeline.set_constant_value("memory", [4, 5])

            with mock.patch.object(pipeline.logger, "info") as info:
                with pipeline.inspect("memory", "first") as resolved:
                    self.assertEqual(resolved.memory, [4, 5])
                    self.assertEqual(resolved.first, [1, 2])

            messages = [call.args[0] for call in info.call_args_list]
            self.assertTrue(
                any(
                    "copy it first with copy_for_inspection" in message
                    for message in messages
                )
            )
            self.assertTrue(
                any(
                    "loaded from disk for this inspection" in message
                    for message in messages
                )
            )

    def test_skipped_object_does_not_repeat_copy_nudge(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = self._pipeline(Path(tmp))
            pipeline.set_constant_value("lock", UncopyableBothWays())
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pipeline.copy_for_inspection(["lock"])

            with mock.patch.object(pipeline.logger, "info") as info:
                with pipeline.inspect("lock") as resolved:
                    self.assertIsInstance(resolved.lock, UncopyableBothWays)

            messages = [call.args[0] for call in info.call_args_list]
            self.assertFalse(
                any("copy it first with copy_for_inspection" in message for message in messages)
            )

    def test_save_pipeline_cleans_inspection_temp_root(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline = self._pipeline(root / "project")
            leftover = root / "project" / "inspection_tmp" / "leftover"
            leftover.mkdir(parents=True)
            (leftover / "file.bin").write_bytes(b"stale")

            pipeline.save_pipeline(root / "bundle")

            self.assertFalse((root / "project" / "inspection_tmp").exists())

    def test_save_pipeline_cleans_child_inspection_temp_roots(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = PipelineHandler("parent", {}, root / "parent")
            child = PipelineHandler("child", {}, root / "child")
            child.add_block("producer", 1).register_function(
                produce_early,
                ["value"],
            )
            parent.add_child_pipeline(child, 1)

            for project_root in (parent.project_root, child.project_root):
                leftover = project_root / "inspection_tmp" / "leftover"
                leftover.mkdir(parents=True)
                (leftover / "file.bin").write_bytes(b"stale")

            parent.save_pipeline(root / "bundle")

            self.assertFalse((parent.project_root / "inspection_tmp").exists())
            self.assertFalse((child.project_root / "inspection_tmp").exists())

    def test_normal_getters_and_execution_bypass_inspection_cache(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = self._pipeline(Path(tmp))
            block = pipeline.add_block("consume", 1)
            block.register_function(
                append_value,
                ["result"],
                param_mapping={"value": "first"},
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pipeline.copy_for_inspection(["first"])
            with pipeline.inspect("first") as resolved:
                resolved.first.append(7)

            self.assertEqual(pipeline.get_constant_value("first"), [1, 2])
            pipeline.run_all()
            self.assertEqual(pipeline.get_value("result"), [1, 2, 99])
            with pipeline.inspect("first") as resolved:
                self.assertEqual(resolved.first, [1, 2, 7])

    def test_new_artifact_generation_bypasses_stale_cache(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = self._pipeline(Path(tmp))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pipeline.copy_for_inspection(["first"])
            stale: list[int] = []
            with pipeline.inspect("first") as resolved:
                stale = resolved.first

            pipeline.set_constant_value("first", [10], to_disk=True)
            with pipeline.inspect("first") as resolved:
                self.assertIsNot(resolved.first, stale)
                self.assertEqual(resolved.first, [10])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pipeline.copy_for_inspection(["first"])
            with pipeline.inspect("first") as resolved:
                self.assertEqual(resolved.first, [10])

    def test_copies_disk_backed_outputs_and_root_storage(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline = PipelineHandler("sources", {}, root / "project")
            block = pipeline.add_block("produce", 1)
            block.register_function(
                produce_output,
                ["output_value"],
                save_to_disk=["output_value"],
            )
            pipeline.run_all()
            hash_id = pipeline.save_to_storage("stored", {"source": "storage"})
            pipeline.update_storage(
                hash_id,
                {"source": "storage"},
                to_disk=True,
            )

            uncached_storage: object = None
            with pipeline.inspect("stored") as resolved:
                uncached_storage = resolved.stored

            with (
                mock.patch.object(
                    ArtifactStore,
                    "load",
                    wraps=pipeline.artifact_store.load,
                ) as load,
                warnings.catch_warnings(),
            ):
                warnings.simplefilter("ignore", UserWarning)
                pipeline.copy_for_inspection(["output_value", "stored"])
                # Only the produced output is an artifact; the stored object is
                # currently loaded in memory and is copied directly.
                self.assertEqual(load.call_count, 1)

                with pipeline.inspect("output_value", "stored") as resolved:
                    self.assertEqual(resolved.output_value, {"source": "output"})
                    self.assertEqual(resolved.stored, uncached_storage)
                self.assertEqual(load.call_count, 1)

            pipeline.unload_inspection_objects()
            with self.assertWarnsRegex(UserWarning, "nothing was unloaded"):
                pipeline.unload_inspection_objects()

    @unittest.skipUnless(find_spec("optuna") is not None, "optuna is not installed")
    def test_disk_backed_optuna_sampler_cannot_bypass_protected_inspection(self) -> None:
        import optuna

        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("sampler-copy", {}, Path(tmp))
            record = pipeline.artifact_store.save(
                "sampler",
                optuna.samplers.RandomSampler(),
                "maker",
                "make",
                "run",
            )
            pipeline.set_constant_value("sampler", record, copy=False)
            block = pipeline.add_block("use", 1)
            block.register_function(
                return_value, ["result"], param_mapping={"value": "sampler"}
            )

            with mock.patch.object(ArtifactStore, "load") as load:
                with self.assertRaisesRegex(ResolutionError, "Optuna"):
                    pipeline.copy_for_inspection(["sampler"])
                load.assert_not_called()
            self.assertEqual(pipeline._inspection_cache_bindings, {})
            with self.assertRaisesRegex(InspectionCopyError, "Optuna"):
                block.inspect(allow_mutable_objects=False)

            # Older pickle records can lack the sampler metadata marker.
            record.metadata.pop("optuna_type", None)
            with self.assertRaisesRegex(ResolutionError, "Optuna"):
                pipeline.copy_for_inspection(["sampler"])
            self.assertEqual(pipeline._inspection_cache_bindings, {})

    def test_copies_config_artifact_records(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline = PipelineHandler("config-source", {}, root / "project")
            config_record = pipeline.artifact_store.save(
                variable_name="frame",
                value={"source": "config"},
                block_name="setup",
                function_name="set_config",
                run_id="run",
            )
            pipeline.set_config("frame", config_record)

            with (
                mock.patch.object(
                    ArtifactStore,
                    "load",
                    wraps=pipeline.artifact_store.load,
                ) as load,
                warnings.catch_warnings(),
            ):
                warnings.simplefilter("ignore", UserWarning)
                pipeline.copy_for_inspection(["frame"])
                self.assertEqual(load.call_count, 1)

                with pipeline.inspect("frame") as resolved:
                    self.assertEqual(resolved.frame, {"source": "config"})
                self.assertEqual(load.call_count, 1)

    def test_cache_is_transient_across_save_and_load(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline = self._pipeline(root / "project")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pipeline.copy_for_inspection(["first"])
            bundle = root / "bundle"
            pipeline.save_pipeline(bundle)

            loaded = PipelineHandler.load_pipeline(
                bundle,
                forced_deleting=True,
                trust_project=True,
            )

            self.assertEqual(loaded._inspection_cache_bindings, {})
            self.assertEqual(loaded._inspection_cache_entries, {})

    def test_memory_delta_logging_uses_process_rss_difference(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = self._pipeline(Path(tmp))
            with (
                mock.patch(
                    "mlpipelineholder.execution.inspection._process_rss_bytes",
                    side_effect=[100, 150],
                ),
                mock.patch.object(pipeline.logger, "info") as info,
                warnings.catch_warnings(),
            ):
                warnings.simplefilter("ignore", UserWarning)
                pipeline.copy_for_inspection(["first"])
            message = next(
                call.args[0]
                for call in info.call_args_list
                if "net process RSS change" in call.args[0]
            )
            self.assertIn(
                "Inspection cache copy net process RSS change (after cleanup): +50 B",
                message,
            )
            self.assertIn("estimated newly cached size:", message)

            with (
                mock.patch(
                    "mlpipelineholder.execution.inspection._process_rss_bytes",
                    side_effect=[150, 120],
                ),
                mock.patch.object(pipeline.logger, "info") as info,
            ):
                pipeline.unload_inspection_objects()
            message = next(
                call.args[0]
                for call in info.call_args_list
                if "net process RSS change" in call.args[0]
            )
            self.assertIn(
                "Inspection cache unload net process RSS change (after cleanup): -30 B",
                message,
            )
            self.assertIn("estimated released cached size:", message)

            with (
                mock.patch(
                    "mlpipelineholder.execution.inspection._process_rss_bytes",
                    side_effect=[200, 170],
                ),
                mock.patch.object(pipeline.logger, "info") as info,
                warnings.catch_warnings(),
            ):
                warnings.simplefilter("ignore", UserWarning)
                pipeline.copy_for_inspection(["second"])
            message = next(
                call.args[0]
                for call in info.call_args_list
                if "net process RSS change" in call.args[0]
            )
            self.assertIn(
                "Inspection cache copy net process RSS change (after cleanup): -30 B",
                message,
            )
            self.assertIn("estimated newly cached size:", message)

    def test_copy_logs_refresh_reminder(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = self._pipeline(Path(tmp))

            with mock.patch.object(pipeline.logger, "info") as info:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    pipeline.copy_for_inspection(["first"])

            messages = [call.args[0] for call in info.call_args_list]
            self.assertTrue(
                any("refresh_copied_objects()" in message for message in messages)
            )

    def test_native_allocator_release_is_best_effort(self) -> None:
        calls: list[str] = []

        class FakePool:
            def release_unused(self) -> None:
                calls.append("release_unused")

        fake_pyarrow = types.SimpleNamespace(default_memory_pool=lambda: FakePool())
        with mock.patch.dict(sys.modules, {"pyarrow": fake_pyarrow}):
            _release_native_allocators()
        self.assertEqual(calls, ["release_unused"])

        class BrokenPool:
            def release_unused(self) -> None:
                raise RuntimeError("pool failure")

        fake_broken = types.SimpleNamespace(default_memory_pool=lambda: BrokenPool())
        with mock.patch.dict(sys.modules, {"pyarrow": fake_broken}):
            _release_native_allocators()

    def test_cache_operations_release_native_allocators(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = self._pipeline(Path(tmp))
            with (
                mock.patch(
                    "mlpipelineholder.execution.inspection._release_native_allocators"
                ) as release,
                warnings.catch_warnings(),
            ):
                warnings.simplefilter("ignore", UserWarning)
                pipeline.copy_for_inspection(["first"])
                self.assertGreaterEqual(release.call_count, 1)

                release.reset_mock()
                pipeline.unload_inspection_objects()

            self.assertGreaterEqual(release.call_count, 2)

    def test_smaps_rollup_rss_parser_uses_current_mapping_total(self) -> None:
        with TemporaryDirectory() as tmp:
            rollup = Path(tmp) / "smaps_rollup"
            rollup.write_text(
                "00400000-7fffffffffff ---p 00000000 00:00 0 [rollup]\n"
                "Rss:             1843200 kB\n"
                "Pss:             1700000 kB\n",
                encoding="utf-8",
            )

            self.assertEqual(
                _read_smaps_rollup_rss(rollup),
                1_843_200 * 1024,
            )

    def test_unload_drops_cache_references_before_final_rss_sample(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("release", {}, Path(tmp))
            pipeline.set_constant_value(
                "payload",
                WeakPayload(1_000_000),
                to_disk=True,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pipeline.copy_for_inspection(["payload"])
            binding = pipeline._inspection_cache_bindings["payload"]
            reference = weakref.ref(
                pipeline._inspection_cache_entries[binding.cache_key].value
            )
            released_at_samples: list[bool] = []

            def rss_sample() -> int:
                released_at_samples.append(reference() is None)
                return 100 if len(released_at_samples) == 1 else 80

            with mock.patch(
                "mlpipelineholder.execution.inspection._process_rss_bytes",
                side_effect=rss_sample,
            ):
                pipeline.unload_inspection_objects()

            self.assertEqual(released_at_samples, [False, True])
            self.assertIsNone(reference())

    def test_unload_reports_copies_still_referenced_elsewhere(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("held-copy", {}, Path(tmp))
            original = WeakPayload(128)
            pipeline.set_constant_value("left", original, copy=False)
            pipeline.set_constant_value("right", original, copy=False)
            pipeline.copy_for_inspection(["left", "right"])
            held: WeakPayload | None = None
            with pipeline.inspect("left") as resolved:
                held = resolved.left
            assert held is not None
            self.assertIsNot(held, original)

            with mock.patch.object(pipeline.logger, "warning") as warning:
                pipeline.unload_inspection_objects()

            self.assertEqual(pipeline._inspection_cache_bindings, {})
            self.assertEqual(pipeline._inspection_cache_entries, {})
            warning.assert_called_once()
            message = warning.call_args.args[0]
            self.assertIn("'left'", message)
            self.assertIn("'right'", message)
            self.assertIn("remain alive outside this cache", message)
            self.assertEqual(held.data, original.data)

    def test_unload_does_not_claim_external_references_without_evidence(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("released-copy", {}, Path(tmp))
            pipeline.set_constant_value("payload", WeakPayload(128), copy=False)
            pipeline.copy_for_inspection(["payload"])
            binding = pipeline._inspection_cache_bindings["payload"]
            reference = weakref.ref(
                pipeline._inspection_cache_entries[binding.cache_key].value
            )
            with mock.patch.object(pipeline.logger, "warning") as warning:
                pipeline.unload_inspection_objects()
            self.assertIsNone(reference())
            warning.assert_not_called()

            pipeline.set_constant_value("items", [1], copy=False)
            pipeline.copy_for_inspection(["items"])
            held_list: list[int] | None = None
            with pipeline.inspect("items") as resolved:
                held_list = resolved.items
            assert held_list is not None
            with mock.patch.object(pipeline.logger, "warning") as warning:
                pipeline.unload_inspection_objects()
            warning.assert_not_called()
            self.assertEqual(held_list, [1])

    def test_retained_estimate_counts_immutables_and_deduplicates_exact_views(
        self,
    ) -> None:
        large_string = "x" * 1_000_000
        copy_estimator = _InspectionMemoryEstimator()
        copy_estimator.add("strings", [large_string])
        retained_estimator = _InspectionMemoryEstimator(retained=True)
        retained_estimator.add("strings", [large_string])

        self.assertLess(copy_estimator.total_bytes, 1_000_000)
        self.assertGreater(retained_estimator.total_bytes, 1_000_000)

        base = np.arange(1_000, dtype=np.int64)
        first = base[:500]
        duplicate = base[:500]
        view_estimator = _InspectionMemoryEstimator(retained=True)
        view_estimator.add("first", first)
        view_estimator.add("duplicate", duplicate)
        self.assertEqual(view_estimator.total_bytes, first.nbytes)

    def test_estimate_failure_does_not_change_successful_cache_copy(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline = self._pipeline(Path(tmp))
            with (
                mock.patch.object(
                    _InspectionMemoryEstimator,
                    "add",
                    side_effect=MemoryError("simulated estimate failure"),
                ),
                mock.patch.object(
                    pipeline,
                    "_attempt_allocator_trim",
                    wraps=pipeline._attempt_allocator_trim,
                ) as trim,
                warnings.catch_warnings(),
            ):
                warnings.simplefilter("ignore", UserWarning)
                pipeline.copy_for_inspection(["first"])

            self.assertIn("first", pipeline._inspection_cache_bindings)
            self.assertGreaterEqual(trim.call_count, 2)

    @unittest.skipUnless(find_spec("torch") is not None, "torch is not installed")
    def test_cuda_estimate_is_reported_separately(self) -> None:
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        tensor = torch.zeros(1_024, device="cuda")
        estimator = _InspectionMemoryEstimator(retained=True)
        estimator.add("tensor", tensor)

        device = 0 if tensor.device.index is None else int(tensor.device.index)
        self.assertEqual(estimator.total_bytes, 0)
        self.assertEqual(
            estimator.cuda_bytes[device],
            tensor.element_size() * tensor.nelement(),
        )
        estimate = InspectionMixin._format_inspection_object_estimate(
            [tensor],
            label="cached size",
        )
        self.assertIn(f"CUDA device {device}", estimate)

    @unittest.skipUnless(
        find_spec("dask.dataframe") is not None,
        "dask is not installed",
    )
    def test_dask_compute_configuration_can_be_replaced(self) -> None:
        import dask.dataframe as dd

        with TemporaryDirectory() as tmp:
            pipeline = PipelineHandler("dask-cache", {}, Path(tmp))
            dask_frame = dd.from_pandas(
                pd.DataFrame({"value": [1, 2, 3]}),
                npartitions=1,
            )
            pipeline.set_constant_value("frame", dask_frame, to_disk=True)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pipeline.copy_for_inspection(["frame"])
            with pipeline.inspect("frame", compute=False) as resolved:
                self.assertIsInstance(resolved.frame, pd.DataFrame)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pipeline.copy_for_inspection(["frame"], compute=False)
            with pipeline.inspect("frame", compute=False) as resolved:
                self.assertIsInstance(resolved.frame, dd.DataFrame)
            with pipeline.inspect("frame", compute=True) as resolved:
                self.assertIsInstance(resolved.frame, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
