from pathlib import Path

import pandas as pd
import pytest
import yaml

from injectr import batch, contract


@pytest.fixture
def base_manifest(tmp_path, make_quiet_npz):
    paths = [str(make_quiet_npz(tmp_path / f"base_{i}.npz", tic_id=1000 + i, seed=i))
             for i in range(3)]
    manifest_path = tmp_path / "base_manifest.csv"
    pd.DataFrame({"path": paths}).to_csv(manifest_path, index=False)
    return manifest_path


@pytest.fixture
def grid_yaml(tmp_path):
    grid = {
        "planet": {"period": [2.0, 8.0], "rp": [0.02, 0.1], "t0": [0.0, 1.0],
                   "a": [10.0, 20.0], "inc": [87.0, 90.0]},
        "eb": {"period": [1.0, 4.0], "rp": [0.05, 0.15], "t0": [0.0, 1.0],
               "a": [6.0, 12.0], "inc": [85.0, 90.0], "secondary_scale": [0.1, 0.4]},
    }
    path = tmp_path / "grid.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(grid, f)
    return path


def test_run_batch_writes_expected_row_count(base_manifest, grid_yaml, tmp_path):
    output_dir = tmp_path / "injected"
    output_manifest = tmp_path / "injection_manifest.csv"
    df = batch.run_batch(
        base_manifest=base_manifest, classes=["planet", "eb"], grid=grid_yaml,
        n_per_class=5, output_dir=output_dir, output_manifest=output_manifest, seed=42,
    )
    assert len(df) == 10
    assert output_manifest.exists()
    for p in df["output_path"]:
        assert Path(p).exists()
        contract.load_base(p)  # every output file is schema-valid


def test_run_batch_is_resumable(base_manifest, grid_yaml, tmp_path):
    output_dir = tmp_path / "injected"
    output_manifest = tmp_path / "injection_manifest.csv"
    df1 = batch.run_batch(
        base_manifest=base_manifest, classes=["planet"], grid=grid_yaml,
        n_per_class=3, output_dir=output_dir, output_manifest=output_manifest, seed=42,
    )
    mtimes_before = {p: Path(p).stat().st_mtime for p in df1["output_path"]}

    df2 = batch.run_batch(
        base_manifest=base_manifest, classes=["planet"], grid=grid_yaml,
        n_per_class=3, output_dir=output_dir, output_manifest=output_manifest, seed=42,
    )
    mtimes_after = {p: Path(p).stat().st_mtime for p in df2["output_path"]}

    assert len(df2) == len(df1)
    assert mtimes_before == mtimes_after  # nothing was rewritten


def test_resumed_run_manifest_matches_file_even_with_different_seed(base_manifest, grid_yaml, tmp_path):
    # Resuming with a DIFFERENT --seed (or grid) than the run that created
    # the file must not draw fresh params and record them in the manifest
    # unused -- the manifest has to describe what's actually in the file.
    output_dir = tmp_path / "injected"
    output_manifest = tmp_path / "injection_manifest.csv"
    df1 = batch.run_batch(
        base_manifest=base_manifest, classes=["planet"], grid=grid_yaml,
        n_per_class=1, output_dir=output_dir, output_manifest=output_manifest, seed=1,
    )

    df2 = batch.run_batch(
        base_manifest=base_manifest, classes=["planet"], grid=grid_yaml,
        n_per_class=1, output_dir=output_dir, output_manifest=output_manifest, seed=999,
    )

    output_path = df2.iloc[0]["output_path"]
    actual_params = contract.load_base(output_path)["injection_params"]

    assert df2.iloc[0]["period"] == pytest.approx(actual_params["period"])
    assert df2.iloc[0]["rp"] == pytest.approx(actual_params["rp"])
    assert df2.iloc[0]["t0"] == pytest.approx(actual_params["t0"])
    # And the file itself was genuinely untouched (still run 1's values).
    assert df2.iloc[0]["period"] == pytest.approx(df1.iloc[0]["period"])


def test_run_batch_unknown_class_raises(base_manifest, grid_yaml, tmp_path):
    with pytest.raises(ValueError):
        batch.run_batch(
            base_manifest=base_manifest, classes=["not_a_class"], grid=grid_yaml,
            n_per_class=1, output_dir=tmp_path / "out", output_manifest=tmp_path / "m.csv",
        )


def test_run_batch_empty_classes_raises(base_manifest, grid_yaml, tmp_path):
    with pytest.raises(ValueError, match="non-empty"):
        batch.run_batch(
            base_manifest=base_manifest, classes=[], grid=grid_yaml,
            n_per_class=1, output_dir=tmp_path / "out", output_manifest=tmp_path / "m.csv",
        )
