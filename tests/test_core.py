import numpy as np
import pytest

from injectr import contract, core


def test_inject_planet_basic(base_npz):
    result = core.inject_planet(base_npz, period=5.2, rp=0.05, t0=0.3, a=15.0, inc=89.0, seed=42)
    assert result.label == "planet"
    assert result.injection_params == {
        "class": "planet", "period": 5.2, "rp": 0.05, "t0": 0.3,
        "a": 15.0, "inc": 89.0, "u": [0.4, 0.25],
    }
    assert result.flux.shape == result.time.shape
    assert result.injected_depth_ppm > 0
    assert result.injected_duration_hours > 0


@pytest.mark.parametrize("bad_period", [0, -5.2])
@pytest.mark.parametrize("inject_fn", [core.inject_planet, core.inject_eb, core.inject_blend])
def test_inject_rejects_non_positive_period(base_npz, inject_fn, bad_period):
    # period<=0 divides by zero / flips sign inside transit_duration_hours,
    # silently producing NaN flux or a nonsensical negative duration
    # instead of a clear error.
    with pytest.raises(ValueError, match="period must be positive"):
        inject_fn(base_npz, period=bad_period, rp=0.05, t0=0.3)


@pytest.mark.parametrize("bad_prot", [0, -1.0])
def test_inject_starspot_rejects_non_positive_prot(base_npz, bad_prot):
    # prot=0 divides by zero in starspot_flux, silently producing all-NaN flux.
    with pytest.raises(ValueError, match="prot must be positive"):
        core.inject_starspot(base_npz, prot=bad_prot)


def test_inject_deterministic_without_extra_noise(base_npz):
    r1 = core.inject_planet(base_npz, period=5.2, rp=0.05, t0=0.3, a=15.0, inc=89.0, seed=42)
    r2 = core.inject_planet(base_npz, period=5.2, rp=0.05, t0=0.3, a=15.0, inc=89.0, seed=42)
    assert np.array_equal(r1.flux, r2.flux)
    assert np.array_equal(r1.time, r2.time)


def test_inject_flux_err_unchanged(base_npz):
    base = np.load(base_npz)
    result = core.inject_planet(base_npz, period=5.2, rp=0.05, t0=0.3, a=15.0, inc=89.0)
    assert np.array_equal(result.flux_err, base["flux_err"])


def test_inject_no_extra_noise_flux_differs_only_by_transit_shape(base_npz):
    base = np.load(base_npz)
    result = core.inject_planet(base_npz, period=5.2, rp=0.05, t0=0.3, a=15.0, inc=89.0)
    # out-of-transit points should be untouched (multiplicative combine, model flux == 1 there)
    out_of_transit = np.abs(result.flux - base["flux"]) < 1e-12
    assert out_of_transit.mean() > 0.9  # most of a sparse-duty-cycle transit is out of transit


def test_to_npz_writes_schema_compliant_file(base_npz, tmp_path):
    result = core.inject_eb(base_npz, period=1.5, rp=0.1, t0=0.2, a=8.0, inc=86.0,
                             secondary_scale=0.3, seed=1)
    out_path = tmp_path / "eb_out.npz"
    result.to_npz(out_path)
    sample = contract.load_base(out_path)
    assert sample["label"] == "eb"
    assert sample["augmented"] == True  # noqa: E712
    assert sample["injection_params"]["class"] == "eb"


def test_inject_blend(base_npz):
    result = core.inject_blend(base_npz, period=4.0, rp=0.08, t0=0.1, a=12.0, inc=88.0,
                                dilution=0.3, seed=7)
    assert result.label == "blend"
    assert result.injection_params["dilution"] == 0.3
    assert result.injected_depth_ppm > 0


def test_inject_starspot(base_npz):
    result = core.inject_starspot(base_npz, prot=6.2, amp1=0.015, amp2=0.004,
                                   phase1=0.5, phase2=1.0, seed=3)
    assert result.label == "starspot"
    assert result.injected_depth_ppm is None
    assert result.injected_duration_hours is None
    assert result.injection_params == {
        "class": "starspot", "prot": 6.2, "amp1": 0.015, "amp2": 0.004,
        "phase1": 0.5, "phase2": 1.0,
    }


def test_extra_noise_ppm_changes_flux_and_is_seed_reproducible(base_npz):
    r1 = core.inject_planet(base_npz, period=5.2, rp=0.05, t0=0.3, seed=42, extra_noise_ppm=50.0)
    r2 = core.inject_planet(base_npz, period=5.2, rp=0.05, t0=0.3, seed=42, extra_noise_ppm=50.0)
    r0 = core.inject_planet(base_npz, period=5.2, rp=0.05, t0=0.3, seed=42, extra_noise_ppm=0.0)
    assert np.array_equal(r1.flux, r2.flux)
    assert not np.array_equal(r1.flux, r0.flux)
