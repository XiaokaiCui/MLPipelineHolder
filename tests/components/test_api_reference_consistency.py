from __future__ import annotations

import html
import inspect
import re
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mlpipelineholder import PipelineHandler
from mlpipelineholder.execution.atom_pipeline import AtomPipeline
from mlpipelineholder.execution.block import ExecutionBlock

_API_REFERENCE = Path(__file__).resolve().parents[2] / "docs" / "api_reference.html"

_METHOD_PATTERN = re.compile(
    r'<h4 class="method-name" id="(?P<id>[^"]+)">.*?</h4>\s*'
    r'<div class="sig">(?P<sig>.*?)</div>',
    re.DOTALL,
)
_PARAMETER_PATTERN = re.compile(r"(?P<name>\*{0,2}[A-Za-z_][A-Za-z0-9_]*)\s*:")

_PUBLIC_METHODS: dict[str, Callable[..., Any]] = {
    "PipelineHolder.inspect": PipelineHandler.inspect,
    "PipelineHolder.inspect_node": PipelineHandler.inspect_node,
    "PipelineHolder.get_selected_node_output": PipelineHandler.get_selected_node_output,
    "PipelineHolder.set_strict_mode": PipelineHandler.set_strict_mode,
    "ExecutionBlock.inspect": ExecutionBlock.inspect,
    "AtomPipeline.inspect": AtomPipeline.inspect,
}


def _documented_signatures() -> dict[str, str]:
    text = _API_REFERENCE.read_text(encoding="utf-8")
    signatures: dict[str, str] = {}
    for match in _METHOD_PATTERN.finditer(text):
        signature = html.unescape(re.sub(r"<[^>]+>", "", match.group("sig")))
        signatures[match.group("id")] = " ".join(signature.split())
    return signatures


def _documented_parameter_segments(signature: str) -> dict[str, str]:
    matches = list(_PARAMETER_PATTERN.finditer(signature))
    segments: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(signature)
        segments[match.group("name").lstrip("*")] = signature[match.start() : end]
    return segments


class ApiReferenceConsistencyTests(unittest.TestCase):
    """Protect key public APIs from drifting away from the committed reference."""

    def test_key_public_methods_are_documented(self) -> None:
        signatures = _documented_signatures()
        for method_id in _PUBLIC_METHODS:
            with self.subTest(method_id=method_id):
                self.assertIn(method_id, signatures)

    def test_documented_parameters_match_runtime_signatures(self) -> None:
        signatures = _documented_signatures()
        for method_id, callable_obj in _PUBLIC_METHODS.items():
            documented = list(_documented_parameter_segments(signatures[method_id]))
            runtime = [
                name
                for name in inspect.signature(callable_obj).parameters
                if name != "self"
            ]
            with self.subTest(method_id=method_id):
                self.assertEqual(documented, runtime)

    def test_documented_defaults_match_runtime_defaults(self) -> None:
        signatures = _documented_signatures()
        for method_id, callable_obj in _PUBLIC_METHODS.items():
            segments = _documented_parameter_segments(signatures[method_id])
            parameters = {
                name: parameter
                for name, parameter in inspect.signature(
                    callable_obj
                ).parameters.items()
                if name != "self"
            }
            for name, parameter in parameters.items():
                with self.subTest(method_id=method_id, parameter=name):
                    if parameter.default is inspect.Parameter.empty:
                        self.assertNotIn("=", segments[name])
                    else:
                        self.assertIn(f"= {parameter.default}", segments[name])


if __name__ == "__main__":
    unittest.main()
