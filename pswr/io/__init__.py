"""I/O submodule (PSWR-1).

Phase 5: YAML/JSON config loader, HDF5/JSON Pareto-result writer.
"""

from .config import (
    load_config,
    config_to_dict,
)
from .results import (
    save_run,
    load_run,
    RunArtifact,
)

__all__ = [
    "load_config", "config_to_dict",
    "save_run", "load_run", "RunArtifact",
]
