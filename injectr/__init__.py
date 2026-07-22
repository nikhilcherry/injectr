"""injectr: inject synthetic transit/EB/blend/starspot signals into real,
already-processed light curves for injection-recovery testing.
"""

from .core import (
    InjectionResult,
    inject_blend,
    inject_eb,
    inject_planet,
    inject_starspot,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "InjectionResult",
    "inject_planet",
    "inject_eb",
    "inject_blend",
    "inject_starspot",
]
