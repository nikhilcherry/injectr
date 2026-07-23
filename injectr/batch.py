"""Grid/manifest-driven batch injection.

Draws `n_per_class` synthetic injections per requested class from a
param-grid YAML, spreads them round-robin across a manifest of base light
curves, and writes each output plus a combined injection_manifest.csv.

Resumable via a simple "does output_path already exist" skip — this is
not batchr's content-hash cache, just enough to make a killed run resume
without redoing finished work. Anyone wanting real parallel/resumable
execution should wrap `injectr.inject_*` with `batchr.run_batch` directly
instead of reimplementing that here (see the README).
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from . import core

INJECT_FNS = {
    "planet": core.inject_planet,
    "eb": core.inject_eb,
    "blend": core.inject_blend,
    "starspot": core.inject_starspot,
}

# Union of columns across all classes; unused columns are left blank/NaN
# for a given row's class.
MANIFEST_COLUMNS = [
    "output_path", "base_path", "label", "seed", "class",
    "period", "rp", "t0", "a", "inc", "ecc", "w", "secondary_scale", "dilution",
    "prot", "amp1", "amp2", "phase1", "phase2",
    "injected_depth_ppm", "injected_duration_hours",
]


def load_base_manifest(path) -> list:
    """Read a CSV of base .npz paths (a 'path' column, or the first column
    if 'path' isn't present)."""
    df = pd.read_csv(path)
    col = "path" if "path" in df.columns else df.columns[0]
    paths = [str(p) for p in df[col].tolist()]
    if not paths:
        raise ValueError(f"{path}: no base file paths found")
    return paths


def load_grid(path) -> dict:
    with open(path) as f:
        grid = yaml.safe_load(f)
    if not isinstance(grid, dict):
        raise ValueError(f"{path}: expected a mapping of class -> param grid")
    return grid


def _draw_param(rng: np.random.Generator, spec: Any):
    """A 2-item numeric list is a uniform range [min, max]; any other list
    is a discrete choice set; a bare scalar is a fixed value."""
    if isinstance(spec, (list, tuple)):
        if len(spec) == 2 and all(isinstance(v, (int, float)) for v in spec):
            lo, hi = spec
            return float(rng.uniform(lo, hi))
        return spec[int(rng.integers(0, len(spec)))]
    return spec


def _draw_params(rng: np.random.Generator, class_grid: dict) -> dict:
    return {name: _draw_param(rng, spec) for name, spec in class_grid.items()}


def run_batch(*, base_manifest, classes, grid, n_per_class, output_dir,
              output_manifest, seed=42, extra_noise_ppm=0.0) -> pd.DataFrame:
    """Draw and write `n_per_class` injections for each of `classes`.

    Returns the injection manifest as a DataFrame (also written to
    `output_manifest`). Any output_path that already exists is skipped
    (not recomputed) but still recorded as a row.
    """
    base_paths = load_base_manifest(base_manifest)
    grid_spec = load_grid(grid)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    unknown = [c for c in classes if c not in INJECT_FNS]
    if unknown:
        raise ValueError(f"unknown injection class(es) {unknown}; choose from {list(INJECT_FNS)}")
    missing_grid = [c for c in classes if c not in grid_spec]
    if missing_grid:
        raise ValueError(f"{grid}: no param grid for class(es) {missing_grid}")

    base_order = list(base_paths)
    np.random.default_rng(seed).shuffle(base_order)
    base_cycle = itertools.cycle(base_order)

    rows = []
    for class_idx, cls in enumerate(classes):
        class_grid = grid_spec[cls]
        for i in range(n_per_class):
            base_path = next(base_cycle)
            draw_seed = int(np.random.SeedSequence([seed, class_idx, i]).generate_state(1)[0])
            rng = np.random.default_rng(draw_seed)
            params = _draw_params(rng, class_grid)

            output_path = output_dir / f"{cls}_{i + 1:04d}.npz"
            row = {"output_path": str(output_path), "base_path": base_path,
                   "label": cls, "seed": draw_seed, "class": cls, **params}

            if output_path.exists():
                rows.append(row)
                continue

            result = INJECT_FNS[cls](base_path, seed=draw_seed,
                                      extra_noise_ppm=extra_noise_ppm, **params)
            result.to_npz(output_path)
            row["injected_depth_ppm"] = result.injected_depth_ppm
            row["injected_duration_hours"] = result.injected_duration_hours
            rows.append(row)

    manifest_df = pd.DataFrame(rows)
    for col in MANIFEST_COLUMNS:
        if col not in manifest_df.columns:
            manifest_df[col] = np.nan
    manifest_df = manifest_df[MANIFEST_COLUMNS]

    output_manifest = Path(output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(output_manifest, index=False)
    return manifest_df
