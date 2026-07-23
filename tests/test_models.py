import numpy as np
import pytest

from injectr import models


def test_planet_depth_matches_analytic_rp_squared():
    time = np.linspace(-0.5, 0.5, 2001)
    rp = 0.05
    flux = models.planet_flux(time, period=1000.0, rp=rp, t0=0.0, a=50.0, inc=90.0, u=[0.0, 0.0])
    depth = 1.0 - flux.min()
    assert depth == pytest.approx(rp ** 2, rel=1e-3)


def test_planet_flux_default_ld_coeffs_is_locked_constant():
    time = np.array([0.0])
    assert models.LD_COEFFS == [0.4, 0.25]
    # default u= None should use LD_COEFFS, same as passing it explicitly
    a = models.planet_flux(time, period=5.0, rp=0.1, t0=0.0, a=15.0, inc=89.0)
    b = models.planet_flux(time, period=5.0, rp=0.1, t0=0.0, a=15.0, inc=89.0, u=models.LD_COEFFS)
    assert np.array_equal(a, b)


def test_eb_secondary_depth_scales_with_secondary_scale():
    period = 2.0
    time = np.linspace(-0.1, 1.1, 4001)
    rp = 0.1
    flux = models.eclipsing_binary_flux(
        time, period=period, rp=rp, t0=0.0, a=50.0, inc=90.0,
        secondary_scale=0.25, u=[0.0, 0.0],
    )
    primary_depth = 1.0 - flux[time < 0.2].min()
    secondary_depth = 1.0 - flux[time > 0.8].min()
    assert primary_depth == pytest.approx(rp ** 2, rel=1e-3)
    assert secondary_depth == pytest.approx(rp ** 2 * 0.25, rel=1e-3)


def test_blend_dilutes_transit_depth():
    time = np.linspace(-0.5, 0.5, 2001)
    rp = 0.1
    dilution = 0.4
    flux = models.blend_flux(time, period=1000.0, rp=rp, t0=0.0, a=50.0, inc=90.0,
                              dilution=dilution, u=[0.0, 0.0])
    depth = 1.0 - flux.min()
    assert depth == pytest.approx(rp ** 2 * dilution, rel=1e-3)


def test_blend_dilution_one_matches_undiluted_planet():
    time = np.linspace(-0.5, 0.5, 2001)
    rp = 0.08
    planet = models.planet_flux(time, period=1000.0, rp=rp, t0=0.0, a=50.0, inc=90.0, u=[0.0, 0.0])
    blended = models.blend_flux(time, period=1000.0, rp=rp, t0=0.0, a=50.0, inc=90.0,
                                 dilution=1.0, u=[0.0, 0.0])
    assert np.allclose(planet, blended)


def test_starspot_single_harmonic_amplitude():
    time = np.linspace(0, 20, 5000)
    flux = models.starspot_flux(time, prot=5.0, amp1=0.02, amp2=0.0)
    assert flux.max() - 1.0 == pytest.approx(0.02, rel=1e-2)
    assert 1.0 - flux.min() == pytest.approx(0.02, rel=1e-2)


def test_transit_duration_hours_positive_and_reasonable():
    duration = models.transit_duration_hours(period=5.2, a=15.0, inc=89.0, rp=0.05)
    assert 0.0 < duration < 10.0


def test_transit_duration_hours_default_ecc_matches_explicit_circular():
    circular = models.transit_duration_hours(period=5.2, a=15.0, inc=89.0, rp=0.05)
    explicit = models.transit_duration_hours(period=5.2, a=15.0, inc=89.0, rp=0.05, ecc=0.0, w=90.0)
    assert circular == pytest.approx(explicit)


def test_transit_duration_hours_eccentric_orbit_correction():
    # Winn 2010 eq. 16: T14_ecc = T14_circ * sqrt(1-e^2) / (1 + e*sin(w)).
    # w=90 (periastron at inferior conjunction) shortens the transit duration.
    circular = models.transit_duration_hours(period=5.2, a=15.0, inc=89.0, rp=0.05)
    eccentric = models.transit_duration_hours(period=5.2, a=15.0, inc=89.0, rp=0.05, ecc=0.3, w=90.0)
    expected = circular * np.sqrt(1.0 - 0.3 ** 2) / (1.0 + 0.3 * np.sin(np.deg2rad(90.0)))
    assert eccentric == pytest.approx(expected)
    assert eccentric < circular
