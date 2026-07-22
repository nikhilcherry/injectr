"""Batman-based forward models for injectr's synthetic signals.

Injection classes and their models:
  - planet: a single transiting planet (batman, quadratic limb darkening).
  - eb: eclipsing binary = a planet-shaped primary plus a secondary
    eclipse at phase 0.5, scaled by `secondary_scale`.
  - blend: a planet transit diluted by a third light source:
    flux = 1 - dilution * (1 - transit_flux).
  - starspot: a two-harmonic sinusoid at Prot and Prot/2 (rotational
    modulation only — no batman, there's no eclipse to model).

batman installs as `batman-package` but imports as `batman`; imported
inside functions so this module (and injectr generally) stays importable
without a compiled batman on the path.
"""

from __future__ import annotations

import numpy as np

LD_COEFFS = [0.4, 0.25]
# Locked to match fitr/models/planet.py's fixed limb-darkening constant
# exactly. Mismatched LD coefficients create a spurious planet/blend
# degeneracy at fit time: blend's extra free `dilution` parameter absorbs
# the shape mismatch and wins on fit quality even though the injected
# truth is `planet` (see arvyo-pipeline/scripts/regenerate_fixtures.py,
# which hit this bug empirically before pinning the same constant).


def planet_flux(time, *, period, rp, t0=0.0, a=15.0, inc=89.0, ecc=0.0, w=90.0, u=None):
    """Transiting-planet flux at `time` (batman, quadratic limb darkening).

    `rp` is Rp/R*, `a` is a/R* (both dimensionless), `inc` in degrees.
    """
    import batman

    bp = batman.TransitParams()
    bp.t0 = t0
    bp.per = period
    bp.rp = rp
    bp.a = a
    bp.inc = inc
    bp.ecc = ecc
    bp.w = w
    bp.u = list(u) if u is not None else list(LD_COEFFS)
    bp.limb_dark = "quadratic"

    model = batman.TransitModel(bp, np.asarray(time, dtype=np.float64))
    return model.light_curve(bp)


def eclipsing_binary_flux(time, *, period, rp, t0=0.0, a=15.0, inc=89.0,
                           secondary_scale=0.1, ecc=0.0, w=90.0, u=None):
    """Primary eclipse (like `planet_flux`) plus a secondary at phase 0.5.

    `secondary_scale` is the secondary depth as a fraction of the primary
    depth; the secondary's radius ratio scales as sqrt of it.
    """
    primary = planet_flux(time, period=period, rp=rp, t0=t0, a=a, inc=inc,
                           ecc=ecc, w=w, u=u)
    secondary = planet_flux(
        time, period=period, rp=rp * np.sqrt(secondary_scale),
        t0=t0 + period / 2.0, a=a, inc=inc, ecc=ecc, w=w, u=u,
    )
    return primary + (secondary - 1.0)


def blend_flux(time, *, period, rp, t0=0.0, a=15.0, inc=89.0, dilution=0.5,
               ecc=0.0, w=90.0, u=None):
    """A planet transit diluted by a third light source.

    flux = 1 - dilution * (1 - transit_flux); dilution=1 recovers the
    undiluted planet transit, dilution -> 0 washes the dip out entirely.
    """
    transit_flux = planet_flux(time, period=period, rp=rp, t0=t0, a=a,
                                inc=inc, ecc=ecc, w=w, u=u)
    return 1.0 - dilution * (1.0 - transit_flux)


def starspot_flux(time, *, prot, amp1=0.01, amp2=0.0, phase1=0.0, phase2=0.0):
    """Sum of 1-2 sinusoids at Prot and Prot/2 (rotational modulation, no eclipse)."""
    time = np.asarray(time, dtype=np.float64)
    flux = 1.0 + amp1 * np.sin(2 * np.pi * time / prot + phase1)
    if amp2:
        flux = flux + amp2 * np.sin(2 * np.pi * time / (prot / 2.0) + phase2)
    return flux


def transit_duration_hours(*, period, a, inc, rp):
    """Analytic first-to-fourth-contact (T14) transit duration, in hours."""
    inc_rad = np.deg2rad(inc)
    b = a * np.cos(inc_rad)
    discriminant = max((1.0 + rp) ** 2 - b ** 2, 0.0)
    arg = np.sqrt(discriminant) / (a * np.sin(inc_rad))
    arg = min(max(arg, -1.0), 1.0)
    duration_days = (period / np.pi) * np.arcsin(arg)
    return duration_days * 24.0
