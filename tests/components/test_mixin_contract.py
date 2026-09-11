from __future__ import annotations

import unittest

from src.mlpipelineholder import PipelineHolder
from src.mlpipelineholder.core.base import PipelineBase

ALLOWED_PUBLIC_COLLISIONS: frozenset[str] = frozenset()


def find_public_collisions(bases: tuple[type, ...]) -> dict[str, list[str]]:
    owners: dict[str, list[str]] = {}
    for base in bases:
        for name, value in vars(base).items():
            if name.startswith("_") or not callable(value):
                continue
            owners.setdefault(name, []).append(base.__name__)
    return {name: names for name, names in owners.items() if len(names) > 1}


class MixinContractTests(unittest.TestCase):
    def test_no_duplicate_public_callables_across_mixins(self) -> None:
        bases = tuple(
            base
            for base in PipelineHolder.__mro__[1:]
            if base not in (object, PipelineBase)
        )
        collisions = find_public_collisions(bases)
        unexpected = {
            name: owners
            for name, owners in collisions.items()
            if name not in ALLOWED_PUBLIC_COLLISIONS
        }
        self.assertEqual(
            unexpected,
            {},
            "Mixins define the same public method name; MRO would silently pick a winner",
        )

    def test_checker_detects_synthetic_duplicate(self) -> None:
        class Alpha:
            def shared(self) -> None: ...

        class Beta:
            def shared(self) -> None: ...

        collisions = find_public_collisions((Alpha, Beta))
        self.assertEqual(collisions, {"shared": ["Alpha", "Beta"]})


if __name__ == "__main__":
    _ = unittest.main()
