"""Command-line interface for injectr: `injectr inject`, `injectr batch`."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__, batch, core

INJECT_FNS = {
    "planet": core.inject_planet,
    "eb": core.inject_eb,
    "blend": core.inject_blend,
    "starspot": core.inject_starspot,
}

# Params required (no CLI default) per class, in the order inject_* expects.
REQUIRED_PARAMS = {
    "planet": ["period", "rp", "t0"],
    "eb": ["period", "rp", "t0"],
    "blend": ["period", "rp", "t0"],
    "starspot": ["prot"],
}

# All params accepted per class (required + optional-with-default).
CLASS_PARAMS = {
    "planet": ["period", "rp", "t0", "a", "inc"],
    "eb": ["period", "rp", "t0", "a", "inc", "secondary_scale"],
    "blend": ["period", "rp", "t0", "a", "inc", "dilution"],
    "starspot": ["prot", "amp1", "amp2", "phase1", "phase2"],
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="injectr",
        description="Inject synthetic transit/EB/blend/starspot signals into real light curves.",
    )
    parser.add_argument("--version", action="version", version=f"injectr {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    inject_p = sub.add_parser("inject", help="Inject a single synthetic signal")
    inject_p.add_argument("base_path")
    inject_p.add_argument("--class", dest="klass", required=True, choices=list(INJECT_FNS))
    inject_p.add_argument("--period", type=float)
    inject_p.add_argument("--rp", type=float)
    inject_p.add_argument("--t0", type=float)
    inject_p.add_argument("--a", type=float, default=15.0)
    inject_p.add_argument("--inc", type=float, default=89.0)
    inject_p.add_argument("--secondary-scale", type=float, default=0.1, dest="secondary_scale")
    inject_p.add_argument("--dilution", type=float, default=0.5)
    inject_p.add_argument("--prot", type=float)
    inject_p.add_argument("--amp1", type=float, default=0.01)
    inject_p.add_argument("--amp2", type=float, default=0.0)
    inject_p.add_argument("--phase1", type=float, default=0.0)
    inject_p.add_argument("--phase2", type=float, default=0.0)
    inject_p.add_argument("--seed", type=int, default=None)
    inject_p.add_argument("--extra-noise-ppm", type=float, default=0.0, dest="extra_noise_ppm")
    inject_p.add_argument("--output", required=True)
    inject_p.add_argument("--json", action="store_true", dest="json_output")

    batch_p = sub.add_parser("batch", help="Batch/sweep injection over a manifest and param grid")
    batch_p.add_argument("--base-manifest", required=True, dest="base_manifest")
    batch_p.add_argument("--classes", required=True,
                          help="Comma-separated class list, e.g. planet,eb,blend")
    batch_p.add_argument("--grid", required=True)
    batch_p.add_argument("--n-per-class", type=int, required=True, dest="n_per_class")
    batch_p.add_argument("--seed", type=int, default=42)
    batch_p.add_argument("--extra-noise-ppm", type=float, default=0.0, dest="extra_noise_ppm")
    batch_p.add_argument("--output-dir", required=True, dest="output_dir")
    batch_p.add_argument("--output-manifest", required=True, dest="output_manifest")

    return parser


def _run_inject(args) -> int:
    fn = INJECT_FNS[args.klass]
    kwargs = {}
    for name in CLASS_PARAMS[args.klass]:
        value = getattr(args, name)
        if value is None and name in REQUIRED_PARAMS[args.klass]:
            flag = "--" + name.replace("_", "-")
            print(f"Error: {flag} is required for --class {args.klass}", file=sys.stderr)
            return 2
        kwargs[name] = value

    result = fn(args.base_path, seed=args.seed, extra_noise_ppm=args.extra_noise_ppm, **kwargs)
    out_path = result.to_npz(args.output)

    if args.json_output:
        print(json.dumps({
            "output_path": str(out_path),
            "label": result.label,
            "injection_params": result.injection_params,
            "injected_depth_ppm": result.injected_depth_ppm,
            "injected_duration_hours": result.injected_duration_hours,
        }))
    else:
        print(f"wrote {out_path} (label={result.label}, "
              f"depth_ppm={result.injected_depth_ppm}, "
              f"duration_hours={result.injected_duration_hours})")
    return 0


def _run_batch(args) -> int:
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    manifest_df = batch.run_batch(
        base_manifest=args.base_manifest, classes=classes, grid=args.grid,
        n_per_class=args.n_per_class, output_dir=args.output_dir,
        output_manifest=args.output_manifest, seed=args.seed,
        extra_noise_ppm=args.extra_noise_ppm,
    )
    print(f"wrote {len(manifest_df)} injection(s) to {args.output_dir}, "
          f"manifest at {args.output_manifest}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inject":
            return _run_inject(args)
        if args.command == "batch":
            return _run_batch(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
