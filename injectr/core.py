"""inject_* functions: load a base light curve, evaluate a forward model on
its own time array, and combine multiplicatively with its own real flux.

The base file's real photometric noise is the noise — no synthetic noise
is added on top by default. `extra_noise_ppm` is an explicit opt-in for
jitter beyond the base file's real noise; it is off (0.0) by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from . import contract, models


@dataclass
class InjectionResult:
    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    label: str
    injection_params: dict
    base_meta: dict = field(repr=False)
    injected_depth_ppm: Optional[float] = None
    injected_duration_hours: Optional[float] = None

    def to_npz(self, path) -> Path:
        return contract.write_injected(
            path,
            time=self.time,
            flux=self.flux,
            flux_err=self.flux_err,
            label=self.label,
            injection_params=self.injection_params,
            base_meta=self.base_meta,
        )


def _apply_extra_noise(flux: np.ndarray, extra_noise_ppm: float, seed) -> np.ndarray:
    if not extra_noise_ppm:
        return flux
    rng = np.random.default_rng(seed)
    jitter = rng.normal(0.0, extra_noise_ppm * 1e-6, size=flux.shape)
    return flux + jitter


def _measured_depth_ppm(model_flux: np.ndarray) -> float:
    return float((1.0 - np.min(model_flux)) * 1e6)


def inject_planet(base_path, period, rp, t0, a=15.0, inc=89.0, ecc=0.0, w=90.0,
                   seed=None, extra_noise_ppm=0.0) -> InjectionResult:
    base = contract.load_base(base_path)
    u = list(models.LD_COEFFS)
    model_flux = models.planet_flux(base["time"], period=period, rp=rp, t0=t0, a=a, inc=inc,
                                     ecc=ecc, w=w, u=u)
    flux = _apply_extra_noise(base["flux"] * model_flux, extra_noise_ppm, seed)

    injection_params = {"class": "planet", "period": period, "rp": rp, "t0": t0,
                         "a": a, "inc": inc, "ecc": ecc, "w": w, "u": u}
    return InjectionResult(
        time=base["time"], flux=flux, flux_err=base["flux_err"], label="planet",
        injection_params=injection_params, base_meta=base,
        injected_depth_ppm=_measured_depth_ppm(model_flux),
        injected_duration_hours=float(
            models.transit_duration_hours(period=period, a=a, inc=inc, rp=rp, ecc=ecc, w=w)
        ),
    )


def inject_eb(base_path, period, rp, t0, a=15.0, inc=89.0, secondary_scale=0.1,
              ecc=0.0, w=90.0, seed=None, extra_noise_ppm=0.0) -> InjectionResult:
    base = contract.load_base(base_path)
    u = list(models.LD_COEFFS)
    model_flux = models.eclipsing_binary_flux(
        base["time"], period=period, rp=rp, t0=t0, a=a, inc=inc,
        secondary_scale=secondary_scale, ecc=ecc, w=w, u=u,
    )
    flux = _apply_extra_noise(base["flux"] * model_flux, extra_noise_ppm, seed)

    injection_params = {"class": "eb", "period": period, "rp": rp, "t0": t0, "a": a,
                         "inc": inc, "secondary_scale": secondary_scale, "ecc": ecc, "w": w, "u": u}
    return InjectionResult(
        time=base["time"], flux=flux, flux_err=base["flux_err"], label="eb",
        injection_params=injection_params, base_meta=base,
        injected_depth_ppm=_measured_depth_ppm(model_flux),
        injected_duration_hours=float(
            models.transit_duration_hours(period=period, a=a, inc=inc, rp=rp, ecc=ecc, w=w)
        ),
    )


def inject_blend(base_path, period, rp, t0, a=15.0, inc=89.0, dilution=0.5,
                  ecc=0.0, w=90.0, seed=None, extra_noise_ppm=0.0) -> InjectionResult:
    base = contract.load_base(base_path)
    u = list(models.LD_COEFFS)
    model_flux = models.blend_flux(
        base["time"], period=period, rp=rp, t0=t0, a=a, inc=inc, dilution=dilution,
        ecc=ecc, w=w, u=u,
    )
    flux = _apply_extra_noise(base["flux"] * model_flux, extra_noise_ppm, seed)

    injection_params = {"class": "blend", "period": period, "rp": rp, "t0": t0, "a": a,
                         "inc": inc, "dilution": dilution, "ecc": ecc, "w": w, "u": u}
    return InjectionResult(
        time=base["time"], flux=flux, flux_err=base["flux_err"], label="blend",
        injection_params=injection_params, base_meta=base,
        injected_depth_ppm=_measured_depth_ppm(model_flux),
        injected_duration_hours=float(
            models.transit_duration_hours(period=period, a=a, inc=inc, rp=rp, ecc=ecc, w=w)
        ),
    )


def inject_starspot(base_path, prot, amp1=0.01, amp2=0.0, phase1=0.0, phase2=0.0,
                     seed=None, extra_noise_ppm=0.0) -> InjectionResult:
    base = contract.load_base(base_path)
    model_flux = models.starspot_flux(
        base["time"], prot=prot, amp1=amp1, amp2=amp2, phase1=phase1, phase2=phase2,
    )
    flux = _apply_extra_noise(base["flux"] * model_flux, extra_noise_ppm, seed)

    injection_params = {"class": "starspot", "prot": prot, "amp1": amp1, "amp2": amp2,
                         "phase1": phase1, "phase2": phase2}
    # No eclipse in this model, so "depth"/"duration" don't apply the way
    # they do for the transit-shaped classes above.
    return InjectionResult(
        time=base["time"], flux=flux, flux_err=base["flux_err"], label="starspot",
        injection_params=injection_params, base_meta=base,
        injected_depth_ppm=None, injected_duration_hours=None,
    )
