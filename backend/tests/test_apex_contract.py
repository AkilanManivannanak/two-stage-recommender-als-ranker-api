"""
test_apex_contract.py — the Salesforce Apex client and the Python GraphQL schema
must agree.

The Apex in salesforce/force-app/.../CineWaveGraphQLClient.cls builds its GraphQL
documents as string literals. Nothing in the Salesforce toolchain checks those
strings against the Python schema, so a field rename here would only surface as
a runtime error in a customer's org.

This test parses the query strings straight out of the .cls source and executes
them against the real schema. If someone renames a field on either side, this
fails in CI rather than in production.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
APEX = ROOT / "salesforce" / "force-app" / "main" / "default" / "classes" / \
       "CineWaveGraphQLClient.cls"
sys.path.insert(0, str(ROOT / "backend" / "src"))


def _apex_string_literal_concat(block: str) -> str:
    """Join an Apex `'a' + 'b' + 'c'` chain into the string it evaluates to."""
    return "".join(re.findall(r"'([^']*)'", block))


def extract_queries() -> dict[str, str]:
    """
    Pull each GraphQL document out of the Apex source by its assignment.

    Apex splices the shared RECO_FIELDS constant into the middle of a string
    concatenation chain, so that identifier is resolved to its own literal value
    BEFORE the chain is joined. Joining first would silently drop it and leave
    an empty `items { }` selection.
    """
    src = APEX.read_text()

    fields_match = re.search(r"RECO_FIELDS\s*=\s*(.*?);", src, re.S)
    reco_fields = _apex_string_literal_concat(fields_match.group(1)) if fields_match else ""

    out: dict[str, str] = {}
    for m in re.finditer(r"String query\s*=\s*(.*?);", src, re.S):
        block = m.group(1)
        # Resolve the constant into an inline literal, then join the chain.
        block = re.sub(r"\bRECO_FIELDS\b", f"'{reco_fields}'", block)
        doc = _apex_string_literal_concat(block)
        if not doc.strip():
            continue
        name = "anonymous"
        for kw in ("query Recs", "query {", "mutation Fb"):
            if doc.strip().startswith(kw):
                name = kw
        out[f"{name}:{len(out)}"] = doc

    out["__RECO_FIELDS__"] = reco_fields
    return out


@pytest.fixture(scope="module")
def schema():
    from recsys.serving.graphql_api import schema as s
    return s


def test_apex_source_is_present():
    assert APEX.exists(), f"Apex client not found at {APEX}"


def test_every_apex_query_validates_against_the_schema(schema):
    from graphql import parse, validate

    queries = {k: v for k, v in extract_queries().items()
               if not k.startswith("__")}
    assert queries, "no GraphQL documents were extracted from the Apex source"

    for name, doc in queries.items():
        errors = validate(schema._schema, parse(doc))
        assert not errors, f"Apex query {name} is invalid against the schema:\n" \
                           f"{doc}\n{[str(e) for e in errors]}"


def test_reco_fields_all_exist_on_the_recommendation_type(schema):
    fields = extract_queries()["__RECO_FIELDS__"].split()
    available = set(schema._schema.type_map["Recommendation"].fields)
    missing = [f for f in fields if f not in available]
    assert not missing, f"Apex requests fields absent from Recommendation: {missing}"


def test_apex_expects_als_and_ranker_scores_separately(schema):
    """
    The Apex asserts these two stay distinct. Guard the schema side too: if the
    two-stage split ever collapses back into one field, this fails here.
    """
    fields = extract_queries()["__RECO_FIELDS__"].split()
    assert "alsScore" in fields and "rankerScore" in fields
    available = schema._schema.type_map["Recommendation"].fields
    assert "alsScore" in available and "rankerScore" in available


def test_apex_queries_execute_against_the_live_resolvers(schema):
    """Validation proves shape; this proves the resolvers actually answer."""
    fields = extract_queries()["__RECO_FIELDS__"]
    doc = ("query { recommendations(userId: 1, k: 3) { modelVersion latencyMs "
           "diversityScore items { " + fields + " } } }")
    result = schema.execute_sync(doc)
    assert result.errors is None, result.errors
    items = result.data["recommendations"]["items"]
    assert len(items) == 3
    assert all(i["title"] for i in items), "titles must be populated"

    result = schema.execute_sync(
        "query { modelMetrics { status ndcgAt10 ndcgAt10AlsOnly "
        "ndcgLiftPctVsAls ipsNdcgAt10 bestBaseline bestBaselineNdcg caveats } }")
    assert result.errors is None, result.errors
    mm = result.data["modelMetrics"]
    assert mm["status"] == "measured", "run the training pipeline before this test"
    assert mm["ndcgAt10"] != 0.1409, "that literal was the fabricated value"

    result = schema.execute_sync(
        'mutation { recordFeedback(userId: 1, itemId: 2858, event: "play", '
        'dwellSeconds: 12.0) { accepted message } }')
    assert result.errors is None, result.errors
    assert result.data["recordFeedback"]["accepted"] is True
