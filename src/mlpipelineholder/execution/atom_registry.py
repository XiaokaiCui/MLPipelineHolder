"""Registry for the atom pipeline class used by holder and registration code."""

from __future__ import annotations

from ..exceptions import RegistrationError

_registered_atom_pipeline_class: type | None = None


def register_atom_pipeline_class(atom_pipeline_class: type) -> None:
    """Record the atom subclass holder code instantiates for atom children."""
    global _registered_atom_pipeline_class
    _registered_atom_pipeline_class = atom_pipeline_class


def atom_pipeline_class() -> type:
    if _registered_atom_pipeline_class is None:
        raise RegistrationError(
            "AtomPipeline is not registered; import mlpipelineholder before creating atoms"
        )
    return _registered_atom_pipeline_class
