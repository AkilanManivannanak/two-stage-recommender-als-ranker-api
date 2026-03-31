import sys
sys.path.insert(0, 'src')
from recsys.serving.catalog_enrichment import enrich_catalog
from recsys.serving.app import CATALOG
enrich_catalog(CATALOG, max_items=500)
print('[llm_enrichment_worker] Done')
