import sys
sys.path.insert(0, 'src')
from recsys.serving.catalog_enrichment import artwork_grounding_audit
from recsys.serving.app import CATALOG
for mid, item in list(CATALOG.items())[:50]:
    r = artwork_grounding_audit(
        item.get('title', ''), item.get('primary_genre', ''),
        item.get('poster_url', ''), {})
    if r.get('trust_score', 1.0) < 0.6:
        print(f'  LOW TRUST: {item["title"]} score={r["trust_score"]}')
print('[vlm_audit_worker] Done')
