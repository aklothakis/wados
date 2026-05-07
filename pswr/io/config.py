"""YAML/JSON config -> :class:`PSWRConfig` mapping (PSWR-1 Phase 5)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict

try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

from ..opt.problem import PSWRConfig


def _read_text(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    text = p.read_text()
    suffix = p.suffix.lower()
    if suffix in (".yaml", ".yml"):
        if not _HAS_YAML:
            raise ImportError("PyYAML required for .yaml configs")
        return _yaml.safe_load(text) or {}
    if suffix == ".json":
        return json.loads(text)
    # Try YAML first, fall back to JSON
    if _HAS_YAML:
        try:
            return _yaml.safe_load(text) or {}
        except Exception:
            pass
    return json.loads(text)


def load_config(path: str | Path) -> PSWRConfig:
    """Read a YAML/JSON file and instantiate a :class:`PSWRConfig`.

    Unknown keys in the file are ignored with a printed warning.
    YAML 1.1 does not parse ``1.0e9`` (without explicit sign) as a number; we
    coerce numeric fields to float/int based on the dataclass field type.
    """
    raw = _read_text(path)
    valid_fields = {f.name: f for f in fields(PSWRConfig)}
    kwargs = {}
    for k, v in raw.items():
        if k in valid_fields:
            kwargs[k] = v
        elif k.startswith("_") or k in ("comment", "description"):
            continue
        else:
            print(f"[pswr.io.config] ignoring unknown key '{k}' in {path}")

    # Type coercion (YAML may return strings for "1.0e9" etc.)
    def _coerce(v, target_type):
        if v is None or isinstance(v, target_type):
            return v
        if target_type is float:
            return float(v)
        if target_type is int:
            try:
                return int(v)
            except (TypeError, ValueError):
                return int(float(v))
        return v

    for name, field_obj in valid_fields.items():
        if name not in kwargs:
            continue
        # Only coerce simple scalars (skip lists, tuples)
        if field_obj.type in (float, "float"):
            kwargs[name] = _coerce(kwargs[name], float)
        elif field_obj.type in (int, "int"):
            kwargs[name] = _coerce(kwargs[name], int)

    if "bistatic_angles_deg" in kwargs:
        kwargs["bistatic_angles_deg"] = [
            tuple(float(v) for v in item)
            for item in kwargs["bistatic_angles_deg"]
        ]
    if "k_i_hat" in kwargs:
        kwargs["k_i_hat"] = tuple(float(v) for v in kwargs["k_i_hat"])
    return PSWRConfig(**kwargs)


def config_to_dict(cfg: PSWRConfig) -> Dict[str, Any]:
    """Round-trip-safe dict (lists for tuples, plain Python scalars) for YAML."""
    d = asdict(cfg)
    # Cast to plain Python types — PyYAML refuses to serialise numpy scalars
    def _py(v):
        if hasattr(v, "item"):
            return v.item()
        return v
    if "bistatic_angles_deg" in d:
        d["bistatic_angles_deg"] = [
            [float(_py(v)) for v in t] for t in d["bistatic_angles_deg"]
        ]
    if "k_i_hat" in d:
        d["k_i_hat"] = [float(_py(v)) for v in d["k_i_hat"]]
    # Coerce numeric scalars
    for k, v in list(d.items()):
        if hasattr(v, "item"):
            d[k] = v.item()
    return d
