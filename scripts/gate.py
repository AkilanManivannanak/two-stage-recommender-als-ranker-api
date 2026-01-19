from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    base = json.loads(Path("reports/ranker_metrics.json").read_text())
    ci = json.loads(Path("reports/ranker_metrics_ci.json").read_text())

    als = base["als_only"]

    # Gate the raw LambdaRank output (keeps CI deltas consistent with your current CI report)
    lgbm = base["als_plus_lambdarank"]

    # Baselines must exist for the baseline-anchored coverage gate
    baselines = base.get("baselines", {})
    if "cooccurrence" not in baselines:
        raise SystemExit("Missing baselines['cooccurrence'] in reports/ranker_metrics.json. Run baselines first.")

    co = baselines["cooccurrence"]

    # Non-regression gates (explicit + defensible)
    # Coverage should not collapse below baseline; require at least 1.2x baseline coverage.
    cov_multiplier = 1.20
    cov_ok = lgbm["coverage10"] >= cov_multiplier * co["coverage10"]

    # Recall@50 shouldn't crater vs ALS-only
    r50_threshold = 0.95
    r50_ok = lgbm["recall50"] >= r50_threshold * als["recall50"]

    # Lift gates using CI lower bounds (statistically defensible)
    d1 = ci["delta"]["lgbm_minus_als_ndcg10"]
    d2 = ci["delta"]["lgbm_minus_co_ndcg10"]
    lift_vs_als_ok = d1["ci95_lo"] > 0.0
    lift_vs_co_ok = d2["ci95_lo"] > 0.0

    result = {
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
