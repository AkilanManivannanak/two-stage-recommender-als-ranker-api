import sys, json
sys.path.insert(0, 'src')
from pathlib import Path
from recsys.serving.agentic_ops import triage_shadow_regression
payload_path = Path('artifacts/bundle/serve_payload.json')
if payload_path.exists():
    payload = json.loads(payload_path.read_text())
    metrics = payload.get('metrics', {})
else:
    metrics = {'ndcg_at_10': 0.1409, 'diversity_score': 0.6923}
baseline = {'ndcg_at_10': 0.0292, 'diversity_score': 0.32}
result = triage_shadow_regression(metrics, baseline)
print(f'[agent_ops] action={result.action} confidence={result.confidence:.2f}')
print(f'[agent_ops] {result.justification}')
print('[agent_ops] requires_human_review=True — no autonomous deployment')
