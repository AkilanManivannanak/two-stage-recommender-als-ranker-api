from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prod", "ci"], default="prod")
    ap.add_argument("--min_users_for_ci_lift", type=int, default=200)
    args = ap.parse_args()

    base = json.loads(Path("reports/ranker_metrics.json").read_text())
    ci = json.loads(Path("reports/ranker_metrics_ci.json").read_text())

    als = base["als_only"]
    lgbm = base["als_plus_lambdarank"]

    baselines = base.get("baselines", {})
    if "cooccurrence" not in baselines:
        raise SystemExit("Missing baselines['cooccurrence'] in reports/ranker_metrics.json. Run baselines first.")

    co = baselines["cooccurrence"]

    # ---------- Non-regression gates ----------
    # Prod gates are stricter; CI gates are looser and meant to catch broken pipelines only.
    if args.mode == "prod":
        cov_multiplier = 1.20
        r50_threshold = 0.95
    else:
        cov_multiplier = 1.00  # must at least match baseline coverage
        r50_threshold = 0.90   # avoid catastrophic recall regressions

    cov_ok = lgbm["coverage10"] >= cov_multiplier * co["coverage10"]
    r50_ok = lgbm["recall50"] >= r50_threshold * als["recall50"]

    # ---------- Lift gates ----------
    d1 = ci["delta"]["lgbm_minus_als_ndcg10"]
    d2 = ci["delta"]["lgbm_minus_co_ndcg10"]

    n_users = int(d1.get("n_users", 0))
    lift_gates_skipped = False
    lift_skip_reason = ""

    if args.mode == "prod":
        # Statistically defensible: CI lower bound must be > 0
        lift_vs_als_ok = float(d1["ci95_lo"]) > 0.0
        lift_vs_co_ok = float(d2["ci95_lo"]) > 0.0
    else:
        # CI mode: small datasets make CI-based lift gates noisy/meaningless.
        # Skip lift checks when too few users.
        if n_users < args.min_users_for_ci_lift:
            lift_gates_skipped = True
            lift_skip_reason = f"n_users={n_users} < {args.min_users_for_ci_lift} (CI lift gates not statistically meaningful)"
            lift_vs_als_ok = True
            lift_vs_co_ok = True
        else:
            # If we have enough users in CI (rare), enforce lower-bound > 0 like prod.
            lift_vs_als_ok = float(d1["ci95_lo"]) > 0.0
            lift_vs_co_ok = float(d2["ci95_lo"]) > 0.0

    result = {
        "mode": args.mode,
        "non_regression": {
            "coverage10_ok": cov_ok,
            "recall50_ok": r50_ok,
            "coverage10_gate": f"coverage10 >= {cov_multiplier:.2f} * cooccurrence_coverage10",
            "coverage10_values": {
                "lgbm_coverage10": float(lgbm["coverage10"]),
                "cooccurrence_coverage10": float(co["coverage10"]),
                "required_min": float(cov_multiplier * co["coverage10"]),
            },
            "recall50_gate": f"recall50 >= {r50_threshold:.2f} * als_recall50",
            "recall50_values": {
                "lgbm_recall50": float(lgbm["recall50"]),
                "als_recall50": float(als["recall50"]),
                "required_min": float(r50_threshold * als["recall50"]),
            },
        },
        "lift": {
            "min_users_for_ci_lift": args.min_users_for_ci_lift,
            "n_users": n_users,
            "lift_gates_skipped": lift_gates_skipped,
            "lift_skip_reason": lift_skip_reason,
            "lgbm_minus_als_ndcg10_ci95_lo": float(d1["ci95_lo"]),
            "lgbm_minus_co_ndcg10_ci95_lo": float(d2["ci95_lo"]),
            "lift_vs_als_ok": lift_vs_als_ok,
            "lift_vs_co_ok": lift_vs_co_ok,
        },
        "pass": bool(cov_ok and r50_ok and lift_vs_als_ok and lift_vs_co_ok),
    }

    Path("reports/gate_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    if not result["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
