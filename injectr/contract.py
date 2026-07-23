"""injectr's own copy of the schema-1.0 constants, plus a light loader for
base light curves and a writer for injected output.

This is a manual sync point with arvyo-pipeline's arvyo/contract.py, not a
shared import: injectr is a standalone, git-installable tool and must not
depend on arvyo-pipeline's internals. If schema-1.0 ever changes, both
copies need to be updated by hand.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "1.0"

REQUIRED_ARRAYS = ["time", "flux", "flux_err"]
OPTIONAL_ARRAYS = ["flux_raw"]
REQUIRED_META = ["tic_id", "label", "sector"]
OPTIONAL_META = ["period_days", "epoch_btjd", "crowdsap", "mission",
                 "augmented", "injection_params"]
LABELS = ["planet", "eb", "blend", "starspot", "null", "unknown"]

# The classes injectr knows how to synthesize and inject.
INJECTABLE_CLASSES = ["planet", "eb", "blend", "starspot"]


class ContractError(ValueError):
    """Raised when a base .npz file violates the schema-1.0 contract."""


def _scalar(value):
    arr = np.asarray(value)
    return arr.item() if arr.shape == () else arr


def load_base(path):
    """Load and validate a base light curve for injection.

    Returns a dict with 'time', 'flux', 'flux_err' arrays, required
    'tic_id'/'label'/'sector' scalars, and any present OPTIONAL_META
    scalars. Raises ContractError if the file doesn't satisfy schema-1.0.
    """
    path = Path(path)
    if not path.exists():
        raise ContractError(f"{path}: file does not exist")
    if path.stat().st_size == 0:
        raise ContractError(f"{path}: file is zero bytes")

    try:
        npz = np.load(path, allow_pickle=True)
    except Exception as exc:
        raise ContractError(f"{path}: could not load npz ({exc})") from exc

    keys = set(npz.files)

    missing_arrays = [k for k in REQUIRED_ARRAYS if k not in keys]
    if missing_arrays:
        raise ContractError(f"{path}: missing required array(s) {missing_arrays}")

    missing_meta = [k for k in REQUIRED_META if k not in keys]
    if missing_meta:
        raise ContractError(f"{path}: missing required meta field(s) {missing_meta}")

    time = np.asarray(npz["time"], dtype=np.float64)
    flux = np.asarray(npz["flux"], dtype=np.float64)
    flux_err = np.asarray(npz["flux_err"], dtype=np.float64)

    if time.ndim != 1:
        raise ContractError(f"{path}: 'time' must be 1D, got shape {time.shape}")
    if flux.shape != time.shape:
        raise ContractError(f"{path}: 'flux' shape {flux.shape} != 'time' shape {time.shape}")
    if flux_err.shape != time.shape:
        raise ContractError(f"{path}: 'flux_err' shape {flux_err.shape} != 'time' shape {time.shape}")

    sample = {"time": time, "flux": flux, "flux_err": flux_err}

    label = str(_scalar(npz["label"]))
    if label not in LABELS:
        raise ContractError(f"{path}: label {label!r} not in {LABELS}")
    sample["label"] = label
    sample["tic_id"] = _scalar(npz["tic_id"])
    sample["sector"] = _scalar(npz["sector"])

    for key in OPTIONAL_META:
        if key in keys:
            sample[key] = _scalar(npz[key])

    return sample


def write_injected(path, *, time, flux, flux_err, label, injection_params, base_meta):
    """Write a schema-1.0-compliant injected .npz file.

    Carries tic_id/sector/mission over from base_meta unchanged; everything
    else (period_days/epoch_btjd/crowdsap/flux_raw) is base-file metadata
    that no longer describes the injected signal, so it is intentionally
    dropped rather than copied.
    """
    if label not in LABELS:
        raise ContractError(f"label {label!r} not in {LABELS}")

    data = {
        "time": np.asarray(time, dtype=np.float64),
        "flux": np.asarray(flux, dtype=np.float64),
        "flux_err": np.asarray(flux_err, dtype=np.float64),
        "tic_id": base_meta["tic_id"],
        "sector": base_meta["sector"],
        "label": label,
        "augmented": True,
        "injection_params": injection_params,
    }
    if "mission" in base_meta:
        data["mission"] = base_meta["mission"]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write (tmp file + rename): np.savez(path, ...) directly would
    # leave a truncated, unreadable .npz behind if the process is killed
    # mid-write -- and since a truncated file still "exists" on disk, a
    # resumed batch run would find it and either crash trying to read it
    # back or (worse) silently treat it as done. A unique-per-call tmp
    # name also means two concurrent writers targeting the same path (two
    # independently invoked batch runs racing on the same output file)
    # don't collide on the same tmp file.
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}-", suffix=".tmp")
    tmp_path = Path(tmp_name)
    # np.savez(path, ...) silently appends ".npz" to any target that
    # doesn't already end in it (so passing tmp_path -- ending in ".tmp"
    # -- would write to a different, wrong filename entirely and leave
    # tmp_path itself as the empty file mkstemp created). Passing the open
    # file object instead avoids that renaming behavior.
    with os.fdopen(fd, "wb") as f:
        np.savez(f, **data)
    os.replace(tmp_path, path)
    return path
