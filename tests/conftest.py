import numpy as np
import pytest


def _make_quiet_npz(path, tic_id, label="null", seed=0):
    rng = np.random.default_rng(seed)
    time = np.arange(0.0, 10.0, 30.0 / 60 / 24)  # 10-day baseline, 30-min cadence
    flux_err_val = 0.0005
    flux = 1.0 + rng.normal(0, flux_err_val, time.size)
    flux_err = np.full(time.size, flux_err_val)
    np.savez(path, time=time, flux=flux, flux_err=flux_err,
              tic_id=tic_id, label=label, sector=5)
    return path


@pytest.fixture
def base_npz(tmp_path):
    return _make_quiet_npz(tmp_path / "base_null.npz", tic_id=123456)


@pytest.fixture
def make_quiet_npz():
    return _make_quiet_npz
