"""I/O submodule (GVWD §5.6).

YAML config loader (per-mode) and JSON+HDF5 result writer with SHA-256
config-hash provenance.
"""

from .config import (
    load_config_yaml,
    config_to_dict,
    EngineeringFlatConfig,
    EngineeringShallowVConfig,
    CaretConfig,
    FlatDeltaConfig,
    MultiWedgeConfig,
    SweepRunConfig,
    config_sha256,
)
from .results import (
    write_run_artifact,
    load_run_artifact,
    RunArtifact,
)

__all__ = [
    "load_config_yaml", "config_to_dict",
    "EngineeringFlatConfig", "EngineeringShallowVConfig",
    "CaretConfig", "FlatDeltaConfig", "MultiWedgeConfig",
    "SweepRunConfig",
    "config_sha256",
    "write_run_artifact", "load_run_artifact", "RunArtifact",
]
