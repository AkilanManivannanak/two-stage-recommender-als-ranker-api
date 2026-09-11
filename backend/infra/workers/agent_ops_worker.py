"""agent_ops_worker.py — periodic agentic triage over the measured metrics."""
from __future__ import annotations
import json, os, sys
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[2]
BUNDLE = Path(os.environ.get("ARTIFACTS_DIR", ROOT / "artifacts")) / "bundle"
sys.path.insert(0, str(ROOT / "src"))

METRICS = BUNDLE / "metrics.json"
if not METRICS.exists():
    sys.exit(f"[agent_ops] no metrics bundle at {METRICS} — run scripts/evaluate.py")

headline = json.loads(METRICS.read_text())["headline"]
from recsys.serving.agentic_ops import triage_shadow_regression
print(json.dumps(triage_shadow_regression(headline), indent=2, default=str))
