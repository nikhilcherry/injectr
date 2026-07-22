import json
import subprocess
import sys

import numpy as np


def _run_cli(args, cwd=None):
    return subprocess.run(
        [sys.executable, "-m", "injectr.cli", *args],
        cwd=cwd, capture_output=True, text=True,
    )


def test_cli_inject_planet_json(base_npz, tmp_path):
    output = tmp_path / "planet_out.npz"
    result = _run_cli([
        "inject", str(base_npz), "--class", "planet",
        "--period", "5.2", "--rp", "0.05", "--t0", "0.3",
        "--seed", "42", "--output", str(output), "--json",
    ])
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["label"] == "planet"
    assert payload["injection_params"]["period"] == 5.2
    assert output.exists()


def test_cli_inject_missing_required_param_errors(base_npz, tmp_path):
    output = tmp_path / "planet_out.npz"
    result = _run_cli([
        "inject", str(base_npz), "--class", "planet",
        "--rp", "0.05", "--t0", "0.3",
        "--output", str(output),
    ])
    assert result.returncode == 2
    assert "--period" in result.stderr


def test_cli_inject_cwd_independence(base_npz, tmp_path):
    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()
    output = tmp_path / "planet_cwd.npz"
    result = _run_cli([
        "inject", str(base_npz), "--class", "planet",
        "--period", "5.2", "--rp", "0.05", "--t0", "0.3",
        "--seed", "42", "--output", str(output), "--json",
    ], cwd=str(other_dir))
    assert result.returncode == 0, result.stderr
    assert output.exists()


def test_cli_batch(base_npz, tmp_path):
    import pandas as pd
    import yaml

    manifest_path = tmp_path / "base_manifest.csv"
    pd.DataFrame({"path": [str(base_npz)]}).to_csv(manifest_path, index=False)

    grid = {"planet": {"period": [2.0, 8.0], "rp": [0.02, 0.1], "t0": [0.0, 1.0]}}
    grid_path = tmp_path / "grid.yaml"
    with open(grid_path, "w") as f:
        yaml.safe_dump(grid, f)

    output_dir = tmp_path / "injected"
    output_manifest = tmp_path / "injection_manifest.csv"
    result = _run_cli([
        "batch", "--base-manifest", str(manifest_path), "--classes", "planet",
        "--grid", str(grid_path), "--n-per-class", "3",
        "--output-dir", str(output_dir), "--output-manifest", str(output_manifest),
    ])
    assert result.returncode == 0, result.stderr
    assert output_manifest.exists()
    df = pd.read_csv(output_manifest)
    assert len(df) == 3
