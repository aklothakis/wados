"""YAML config loading + per-mode dataclass containers (GVWD §5.6).

Each mode has its own config dataclass mirroring the inputs of the
corresponding geometry generator. The CLI examples build an instance
directly or load it from a YAML file via :func:`load_config_yaml`.

The ``mode`` field tags which dataclass type to instantiate when loading
a generic config dict.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


# ----------------------------------------------------------------------
#  Per-mode dataclass containers
# ----------------------------------------------------------------------

@dataclass
class EngineeringFlatConfig:
    mode: str = "engineering_flat"
    M_design: float = 15.0
    theta_fore_deg: float = 8.0
    Lambda_deg: float = 75.0
    L_fore: float = 2.5
    L_center: float = 1.5
    b_base: float = 0.5
    h_base: float = 0.4
    h_fore_nose: float = 0.0
    r_LE_mm: float = 5.0
    r_nose_mm: float = 10.0
    theta_upper_deg: float = 0.0
    gamma: float = 1.4


@dataclass
class EngineeringShallowVConfig:
    mode: str = "engineering_shallow_v"
    M_design: float = 15.0
    theta_fore_deg: float = 8.0
    Lambda_deg: float = 75.0
    L_fore: float = 2.5
    L_center: float = 1.5
    b_base: float = 0.5
    h_base: float = 0.4
    h_fore_nose: float = 0.0
    r_LE_mm: float = 5.0
    r_nose_mm: float = 10.0
    theta_upper_deg: float = 0.0
    dihedral_lower_deg: float = 5.0
    gamma: float = 1.4


@dataclass
class CaretConfig:
    mode: str = "caret"
    M_design: float = 6.0
    theta_d_deg: float = 14.0
    Lambda_deg: float = 70.0
    L: float = 10.0
    gamma: float = 1.4


@dataclass
class FlatDeltaConfig:
    mode: str = "flat_delta"
    M_design: float = 5.0
    theta_d_deg: float = 12.0
    Lambda_deg: float = 75.0
    L: float = 8.0
    gamma: float = 1.4


@dataclass
class MultiWedgeConfig:
    mode: str = "multi_wedge"
    M_design: float = 5.0
    n: int = 2
    delta_total_deg: float = 20.8
    L: float = 8.0
    half_span: float = 1.0
    extrusion: str = "rectangular"
    height: float = 0.6
    gamma: float = 1.4


@dataclass
class FinsConfig:
    n_fins: int = 0
    root_chord: float = 0.3
    tip_chord: float = 0.1
    span: float = 0.4
    sweep_LE_deg: float = 45.0
    dihedral_deg: float = 45.0
    t_c: float = 0.05
    max_thickness_loc: float = 0.5
    LE_style: str = "blunt_cylinder"
    LE_radius_mm: float = 1.0
    attach_x_frac: float = 0.5


@dataclass
class SweepRunConfig:
    """Optional sweep settings (off-design Mach-alpha grid)."""
    enabled: bool = False
    M_grid: Tuple[float, float, int] = (5.0, 20.0, 8)
    alpha_grid_deg: Tuple[float, float, int] = (0.0, 15.0, 6)
    altitude_km: float = 30.0
    T_w: float = 1500.0
    Re_x_tr: float = 1.0e6


@dataclass
class GVWDRunConfig:
    """Top-level container linking a geometry config + optional fins +
    optional sweep into one run."""
    geometry: Any = None
    fins: FinsConfig = field(default_factory=FinsConfig)
    sweep: SweepRunConfig = field(default_factory=SweepRunConfig)
    output_dir: str = "results"
    tag: str = ""


# ----------------------------------------------------------------------
#  YAML / JSON loading
# ----------------------------------------------------------------------

_MODE_TO_CLS = {
    "engineering_flat": EngineeringFlatConfig,
    "engineering_shallow_v": EngineeringShallowVConfig,
    "caret": CaretConfig,
    "flat_delta": FlatDeltaConfig,
    "multi_wedge": MultiWedgeConfig,
}


def _read_text(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    text = p.read_text()
    suf = p.suffix.lower()
    if suf in (".yaml", ".yml"):
        if not _HAS_YAML:
            raise ImportError("PyYAML required for .yaml configs")
        return _yaml.safe_load(text) or {}
    if suf == ".json":
        return json.loads(text)
    if _HAS_YAML:
        try:
            return _yaml.safe_load(text) or {}
        except Exception:
            pass
    return json.loads(text)


def _coerce_numerics(d: dict, target_cls) -> dict:
    """Coerce strings produced by YAML 1.1 (e.g. ``1.0e9`` parses to str)
    back to numerics based on dataclass field types."""
    out = dict(d)
    for f in fields(target_cls):
        if f.name not in out or out[f.name] is None:
            continue
        v = out[f.name]
        try:
            if f.type in (float, "float") and not isinstance(v, float):
                out[f.name] = float(v)
            elif f.type in (int, "int") and not isinstance(v, int):
                out[f.name] = int(v)
        except (TypeError, ValueError):
            pass
    return out


def load_config_yaml(path: str | Path) -> GVWDRunConfig:
    """Read a YAML / JSON file and return a :class:`GVWDRunConfig`.

    Expected top-level structure::

        geometry:
          mode: engineering_flat        # or caret, flat_delta, ...
          M_design: 15
          theta_fore_deg: 8
          ...
        fins:                            # optional
          n_fins: 4
          ...
        sweep:                           # optional
          enabled: true
          M_grid: [5, 20, 8]
          alpha_grid_deg: [0, 15, 6]
        output_dir: results
        tag: htv2_demo
    """
    raw = _read_text(path)
    geom_d = raw.get("geometry", {})
    mode = geom_d.get("mode", "engineering_flat")
    if mode not in _MODE_TO_CLS:
        raise ValueError(f"unknown mode {mode!r}; valid: {list(_MODE_TO_CLS)}")
    GeomCls = _MODE_TO_CLS[mode]
    geom_d_clean = _coerce_numerics(geom_d, GeomCls)
    # Drop unknown keys
    valid_geom = {f.name for f in fields(GeomCls)}
    geom_d_clean = {k: v for k, v in geom_d_clean.items() if k in valid_geom}
    geom_cfg = GeomCls(**geom_d_clean)

    fins_d = raw.get("fins", {}) or {}
    fins_d = _coerce_numerics(fins_d, FinsConfig)
    valid_fins = {f.name for f in fields(FinsConfig)}
    fins_cfg = FinsConfig(**{k: v for k, v in fins_d.items() if k in valid_fins})

    sweep_d = raw.get("sweep", {}) or {}
    if "M_grid" in sweep_d:
        sweep_d["M_grid"] = tuple(sweep_d["M_grid"])
    if "alpha_grid_deg" in sweep_d:
        sweep_d["alpha_grid_deg"] = tuple(sweep_d["alpha_grid_deg"])
    sweep_d = _coerce_numerics(sweep_d, SweepRunConfig)
    valid_sw = {f.name for f in fields(SweepRunConfig)}
    sweep_cfg = SweepRunConfig(**{k: v for k, v in sweep_d.items()
                                    if k in valid_sw})

    return GVWDRunConfig(
        geometry=geom_cfg, fins=fins_cfg, sweep=sweep_cfg,
        output_dir=str(raw.get("output_dir", "results")),
        tag=str(raw.get("tag", "")),
    )


def config_to_dict(cfg: GVWDRunConfig) -> Dict[str, Any]:
    """Round-trippable dict serialisation."""
    return {
        "geometry": asdict(cfg.geometry),
        "fins": asdict(cfg.fins),
        "sweep": _sweep_to_dict(cfg.sweep),
        "output_dir": cfg.output_dir,
        "tag": cfg.tag,
    }


def _sweep_to_dict(s: SweepRunConfig) -> dict:
    d = asdict(s)
    if "M_grid" in d:
        d["M_grid"] = list(d["M_grid"])
    if "alpha_grid_deg" in d:
        d["alpha_grid_deg"] = list(d["alpha_grid_deg"])
    return d


def config_sha256(cfg: GVWDRunConfig) -> str:
    """Stable SHA-256 hash of a GVWDRunConfig.

    Used as a provenance tag in result artifacts: rerunning with the
    same YAML must produce the same hash, allowing byte-comparable
    output reproduction.
    """
    payload = json.dumps(config_to_dict(cfg), sort_keys=True,
                          separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_geometry(cfg: GVWDRunConfig):
    """Instantiate the geometry object specified by ``cfg.geometry``."""
    from gvwd.geometry import (
        Caret, FlatDelta, MultiWedge, EngineeringFlat, EngineeringShallowV,
    )
    g = cfg.geometry
    if isinstance(g, EngineeringFlatConfig):
        return EngineeringFlat(
            M_design=g.M_design,
            theta_fore=math.radians(g.theta_fore_deg),
            Lambda=math.radians(g.Lambda_deg),
            L_fore=g.L_fore, L_center=g.L_center,
            b_base=g.b_base, h_base=g.h_base,
            h_fore_nose=g.h_fore_nose,
            r_LE=g.r_LE_mm * 1e-3, r_nose=g.r_nose_mm * 1e-3,
            theta_upper=math.radians(g.theta_upper_deg),
            gamma=g.gamma,
        )
    if isinstance(g, EngineeringShallowVConfig):
        return EngineeringShallowV(
            M_design=g.M_design,
            theta_fore=math.radians(g.theta_fore_deg),
            Lambda=math.radians(g.Lambda_deg),
            L_fore=g.L_fore, L_center=g.L_center,
            b_base=g.b_base, h_base=g.h_base,
            h_fore_nose=g.h_fore_nose,
            r_LE=g.r_LE_mm * 1e-3, r_nose=g.r_nose_mm * 1e-3,
            theta_upper=math.radians(g.theta_upper_deg),
            dihedral_lower=math.radians(g.dihedral_lower_deg),
            gamma=g.gamma,
        )
    if isinstance(g, CaretConfig):
        return Caret(
            M_design=g.M_design, theta_d=math.radians(g.theta_d_deg),
            Lambda=math.radians(g.Lambda_deg), L=g.L, gamma=g.gamma,
        )
    if isinstance(g, FlatDeltaConfig):
        return FlatDelta(
            M_design=g.M_design, theta_d=math.radians(g.theta_d_deg),
            Lambda=math.radians(g.Lambda_deg), L=g.L, gamma=g.gamma,
        )
    if isinstance(g, MultiWedgeConfig):
        return MultiWedge(
            M_design=g.M_design, n=g.n,
            delta_total_deg=g.delta_total_deg, L=g.L,
            half_span=g.half_span, extrusion=g.extrusion,
            height=g.height, gamma=g.gamma,
        )
    raise ValueError(f"unknown geometry config type {type(g).__name__}")
