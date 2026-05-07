"""Persistent run artifact: HDF5 Pareto + JSON summary + config dump.

Layout per spec §9:

    results/run_<timestamp>/
        config.yaml                   # exact config used
        pareto.h5                     # all non-dominated solutions
        pareto.json                   # human-readable summary
        plots/  (populated by viz)
        log.txt
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    import h5py
    _HAS_H5 = True
except ImportError:
    _HAS_H5 = False

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

from ..opt.problem import PSWRConfig
from ..opt.run import ParetoResult


def _timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%S")


@dataclass
class RunArtifact:
    """Bundle of inputs + outputs for a complete run."""
    cfg: PSWRConfig
    result: ParetoResult
    base_dir: Path

    @property
    def plots_dir(self) -> Path:
        return self.base_dir / "plots"


def _write_yaml(path: Path, data: Dict[str, Any]) -> None:
    if _HAS_YAML:
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)
    else:
        # Fallback: write JSON with .yaml extension (still loadable by load_config)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def save_run(cfg: PSWRConfig, result: ParetoResult, *,
             out_root: str | Path = "results",
             tag: str = "") -> RunArtifact:
    """Persist the run to ``results/run_<timestamp>[_tag]/``."""
    from .config import config_to_dict
    ts = _timestamp()
    name = f"run_{ts}" + (f"_{tag}" if tag else "")
    base = Path(out_root) / name
    base.mkdir(parents=True, exist_ok=True)
    (base / "plots").mkdir(exist_ok=True)

    # Config
    _write_yaml(base / "config.yaml", config_to_dict(cfg))

    # JSON summary
    summary = {
        "n_eval": int(result.n_eval),
        "n_feasible": int(result.n_feasible),
        "n_pareto": int(result.X.shape[0]),
        "wall_time_s": float(result.wall_time_s),
        "pop_size": int(result.pop_size),
        "n_gen": int(result.n_gen),
        "seed": int(result.seed),
        "ranges": {
            "LD":   [float(-result.F[:, 0].max()), float(-result.F[:, 0].min())] if result.F.size else None,
            "sigma_dBsm": [float(result.F[:, 1].min()), float(result.F[:, 1].max())] if result.F.size else None,
            "eta_V": [float(-result.F[:, 2].max()), float(-result.F[:, 2].min())] if result.F.size else None,
        },
    }
    with open(base / "pareto.json", "w") as f:
        json.dump(summary, f, indent=2)

    # HDF5 Pareto data
    if _HAS_H5:
        with h5py.File(base / "pareto.h5", "w") as h:
            h.create_dataset("X", data=result.X)
            h.create_dataset("F", data=result.F)
            h.create_dataset("G", data=result.G)
            h.attrs["pop_size"] = result.pop_size
            h.attrs["n_gen"] = result.n_gen
            h.attrs["seed"] = result.seed
            h.attrs["wall_time_s"] = result.wall_time_s
            h.attrs["n_eval"] = result.n_eval
            h.attrs["n_feasible"] = result.n_feasible
            if result.history_F:
                gh = h.create_group("history")
                for i, F_gen in enumerate(result.history_F):
                    gh.create_dataset(f"gen_{i:04d}", data=F_gen)
    else:
        # Fallback: pickled .pkl
        import pickle
        with open(base / "pareto.pkl", "wb") as f:
            pickle.dump({"X": result.X, "F": result.F, "G": result.G,
                         "history_F": result.history_F}, f)

    # Log
    with open(base / "log.txt", "w") as f:
        f.write(f"PSWR-1 run {ts}\n")
        f.write(f"pop={result.pop_size} n_gen={result.n_gen} seed={result.seed}\n")
        f.write(f"wall_time = {result.wall_time_s:.2f} s\n")
        f.write(f"n_eval={result.n_eval} n_feasible={result.n_feasible}\n")
        f.write(f"n_pareto={result.X.shape[0]}\n")

    return RunArtifact(cfg=cfg, result=result, base_dir=base)


def load_run(base_dir: str | Path) -> RunArtifact:
    base = Path(base_dir)
    from .config import load_config
    cfg = load_config(base / "config.yaml")

    if _HAS_H5 and (base / "pareto.h5").exists():
        with h5py.File(base / "pareto.h5", "r") as h:
            X = h["X"][...]
            F = h["F"][...]
            G = h["G"][...]
            attrs = dict(h.attrs)
            history_F = []
            if "history" in h:
                gh = h["history"]
                for k in sorted(gh.keys()):
                    history_F.append(gh[k][...])
        result = ParetoResult(
            X=X, F=F, G=G,
            n_eval=int(attrs.get("n_eval", 0)),
            n_feasible=int(attrs.get("n_feasible", 0)),
            wall_time_s=float(attrs.get("wall_time_s", 0.0)),
            history_F=history_F,
            pop_size=int(attrs.get("pop_size", 0)),
            n_gen=int(attrs.get("n_gen", 0)),
            seed=int(attrs.get("seed", 0)),
        )
    else:
        import pickle
        with open(base / "pareto.pkl", "rb") as f:
            d = pickle.load(f)
        result = ParetoResult(
            X=d["X"], F=d["F"], G=d["G"],
            n_eval=0, n_feasible=0, wall_time_s=0.0,
            history_F=d.get("history_F", []),
            pop_size=0, n_gen=0, seed=0,
        )
    return RunArtifact(cfg=cfg, result=result, base_dir=base)
