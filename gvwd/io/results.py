"""Run-artifact writer/reader (GVWD §5.6, §7).

Layout per spec §7::

    results/gvwd_<timestamp>[_tag]/
        config.yaml
        config_sha256.txt
        geometry.stl
        geometry.step                (optional)
        coefficients_on_design.json
        volumetric.json
        heating.json
        sweep_results.h5             (only if sweep was run)
        plots/                        (populated by viz)
        log.txt
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    import h5py
    _HAS_H5 = True
except ImportError:
    _HAS_H5 = False

try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

from .config import (
    GVWDRunConfig, config_to_dict, config_sha256,
)


@dataclass
class RunArtifact:
    """Bundle of everything written for a single run."""
    base_dir: Path
    cfg: GVWDRunConfig
    sha256: str
    on_design: Optional[Dict[str, Any]] = None
    sweep_df: Any = None    # Optional[pandas.DataFrame]


def _write_yaml(path: Path, data: Dict[str, Any]) -> None:
    if _HAS_YAML:
        with open(path, "w") as f:
            _yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)
    else:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def _ts() -> str:
    return time.strftime("%Y%m%dT%H%M%S")


def write_run_artifact(cfg: GVWDRunConfig,
                        on_design: Optional[Dict[str, Any]] = None,
                        sweep_df=None,
                        mesh=None,
                        *,
                        out_root: Optional[str | Path] = None,
                        write_step: bool = False,
                        write_iges: bool = False) -> RunArtifact:
    """Persist a complete run to disk.

    Parameters
    ----------
    cfg          : GVWDRunConfig
    on_design    : dict from aero_coefficients_full (optional)
    sweep_df     : pandas DataFrame from mach_alpha_sweep (optional)
    mesh         : Mesh to export as STL/STEP/IGES (optional)
    out_root     : root directory for ``gvwd_<timestamp>_<tag>/``. If
                   None, uses ``cfg.output_dir``.
    write_step   : also export geometry as STEP (requires cadquery)
    write_iges   : also export geometry as IGES (requires cadquery)
    """
    sha = config_sha256(cfg)
    root = Path(out_root) if out_root is not None else Path(cfg.output_dir)
    name = f"gvwd_{_ts()}" + (f"_{cfg.tag}" if cfg.tag else "")
    base = root / name
    (base / "plots").mkdir(parents=True, exist_ok=True)

    # Config + provenance
    _write_yaml(base / "config.yaml", config_to_dict(cfg))
    (base / "config_sha256.txt").write_text(sha + "\n")

    # On-design coefficients
    if on_design is not None:
        _write_json_with_meta(base / "coefficients_on_design.json",
                                on_design, sha)

    # Sweep DataFrame
    if sweep_df is not None and _HAS_H5:
        with h5py.File(base / "sweep_results.h5", "w") as h:
            for col in sweep_df.columns:
                h.create_dataset(col, data=np.asarray(sweep_df[col]))
            for k, v in (sweep_df.attrs.items() if hasattr(sweep_df, "attrs") else []):
                if isinstance(v, (int, float, str)):
                    h.attrs[k] = v
            h.attrs["config_sha256"] = sha
            h.attrs["n_rows"] = len(sweep_df)
        # Also a JSON copy for human inspection
        sweep_df.to_json(base / "sweep_results.json", orient="records",
                          indent=2)
    elif sweep_df is not None:
        sweep_df.to_json(base / "sweep_results.json", orient="records",
                          indent=2)

    # Mesh exports
    if mesh is not None:
        from gvwd.export.stl import write_stl
        write_stl(mesh, base / "geometry.stl",
                    header=f"gvwd {sha[:12]} {cfg.tag or 'untagged'}")
        if write_step:
            try:
                from gvwd.export.step import write_step as _ws
                _ws(mesh, base / "geometry.step")
            except Exception as e:
                (base / "geometry_step_failed.txt").write_text(str(e))
        if write_iges:
            try:
                from gvwd.export.iges import write_iges as _wi
                _wi(mesh, base / "geometry.iges")
            except Exception as e:
                (base / "geometry_iges_failed.txt").write_text(str(e))

    # Geometric quantities
    if mesh is not None:
        from gvwd.geometry import numerical_volume, planform_area_from_mesh, eta_V
        V = numerical_volume(mesh)
        S = planform_area_from_mesh(mesh)
        e = eta_V(V, S)
        _write_json_with_meta(base / "volumetric.json",
                                {"V_m3": V, "S_planform_m2": S, "eta_V": e},
                                sha)

    # Log
    with open(base / "log.txt", "w") as f:
        f.write(f"GVWD run {name}\nSHA-256: {sha}\n")
        f.write(f"Geometry mode: {cfg.geometry.mode if cfg.geometry else '?'}\n")
        f.write(f"Sweep enabled: {cfg.sweep.enabled}\n")
        if mesh is not None:
            f.write(f"Mesh: {mesh.n_vertices} vertices, "
                     f"{mesh.n_faces} faces\n")

    return RunArtifact(base_dir=base, cfg=cfg, sha256=sha,
                         on_design=on_design, sweep_df=sweep_df)


def _write_json_with_meta(path: Path, payload: Dict[str, Any],
                            sha: str) -> None:
    """JSON writer that strips numpy types and embeds the config hash."""
    def _clean(o):
        if hasattr(o, "item"):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, dict):
            return {str(k): _clean(v) for k, v in o.items()
                     if not k.startswith("_") and not _is_callable(v)}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        return o
    out = {
        "gvwd_config_sha256": sha,
        "data": _clean(payload),
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)


def _is_callable(v) -> bool:
    return callable(v) and not isinstance(v, (np.ndarray, dict, list, tuple))


def load_run_artifact(base_dir: str | Path) -> RunArtifact:
    """Read back a previously-saved run."""
    from .config import load_config_yaml
    base = Path(base_dir)
    cfg = load_config_yaml(base / "config.yaml")
    sha = (base / "config_sha256.txt").read_text().strip() if (base / "config_sha256.txt").exists() else ""
    on_design = None
    if (base / "coefficients_on_design.json").exists():
        on_design = json.loads((base / "coefficients_on_design.json").read_text())
    sweep_df = None
    if (base / "sweep_results.h5").exists() and _HAS_H5:
        import pandas as pd
        with h5py.File(base / "sweep_results.h5", "r") as h:
            cols = {k: h[k][...] for k in h.keys()}
        sweep_df = pd.DataFrame(cols)
    elif (base / "sweep_results.json").exists():
        import pandas as pd
        sweep_df = pd.read_json(base / "sweep_results.json")
    return RunArtifact(base_dir=base, cfg=cfg, sha256=sha,
                         on_design=on_design, sweep_df=sweep_df)
