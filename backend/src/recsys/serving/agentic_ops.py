"""
Agentic Evaluation & Operations Layer
======================================
Plane: Agentic Eval/Ops (NOT in the request path)

This is where agentic AI is actually useful in a recommender system:
  - Orchestrating offline experiments (not deploying them autonomously)
  - Summarising shadow-run regressions in natural language
  - Investigating drift across slices and naming likely causes
  - Generating rollback recommendations with supporting evidence
  - Running policy compliance checks before release
  - Helping engineers understand which component failed and why

What agents do NOT do here:
  - They do not deploy models autonomously
  - They do not rewrite ranking policy
  - They do not override release gates
  - They do not sit inside the latency-critical request loop

The OpenAI Agents SDK pattern: tools + structured outputs + guardrails + tracing.
Here we implement the same pattern using direct API calls with tool definitions,
structured outputs, and an explicit human-review gate before any action.

Reference: OpenAI Agents SDK — tool-enabled operational workflows with tracing
"""
from __future__ import annotations

import http.client, json, os, ssl, time
from dataclasses import dataclass, field
from typing import Any

_OAI_KEY = os.environ.get("OPENAI_API_KEY","")


# ── Agent tools (callable, audited, constrained) ─────────────────────
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "summarise_shadow_regression",
            "description": "Summarise a shadow deployment result comparing new model vs baseline",
            "parameters": {
                "type": "object",
                "properties": {
                    "new_metrics":      {"type":"object"},
                    "baseline_metrics": {"type":"object"},
                    "n_users":          {"type":"integer"},
                },
                "required": ["new_metrics","baseline_metrics"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "investigate_drift",
            "description": "Investigate a data drift report and suggest likely causes",
            "parameters": {
                "type": "object",
                "properties": {
                    "drift_report":  {"type":"object"},
                    "recent_events": {"type":"array","items":{"type":"string"}},
                },
                "required": ["drift_report"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_rollback_decision",
            "description": "Given eval metrics, recommend DEPLOY / HOLD / ROLLBACK with justification",
            "parameters": {
                "type": "object",
                "properties": {
                    "metrics":     {"type":"object"},
                    "gate_result": {"type":"object"},
                    "thresholds":  {"type":"object"},
                },
                "required": ["metrics","gate_result"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "policy_compliance_check",
            "description": "Check model artifacts for policy and safety compliance before release",
            "parameters": {
                "type": "object",
                "properties": {
                    "artwork_audit_results": {"type":"array"},
                    "explanation_samples":   {"type":"array"},
                    "diversity_metrics":     {"type":"object"},
                },
                "required": ["diversity_metrics"],
            },
        },
    },
]


@dataclass
class AgentResult:
    action:       str            # what the agent recommends
    justification:str            # why
    confidence:   float          # 0–1
    requires_human_review: bool  # always True for deploy/rollback
    tool_calls:   list[dict] = field(default_factory=list)
    raw_response: str = ""


def _call_openai_with_tools(system: str, user: str,
                             tools: list[dict]) -> dict:
    """Call OpenAI with tool definitions and return the response."""
    if not _OAI_KEY:
        return {"choices":[{"message":{
            "content": json.dumps({"action":"HOLD","justification":"OpenAI not configured",
                                   "confidence":0.5,"requires_human_review":True})
        }}]}
    body = json.dumps({
        "model": "gpt-4o-mini",
        "messages": [
            {"role":"system","content":system},
            {"role":"user",  "content":user},
        ],
        "tools":       tools,
        "tool_choice": "auto",
        "max_tokens":  600,
        "temperature": 0.2,
        "response_format": {"type":"json_object"},
    }, ensure_ascii=True).encode("utf-8")
    try:
        ctx  = ssl.create_default_context()
        conn = http.client.HTTPSConnection("api.openai.com", timeout=12, context=ctx)
        conn.request("POST","/v1/chat/completions",body=body,headers={
            "Content-Type":"application/json",
            "Authorization":f"Bearer {_OAI_KEY}",
            "Content-Length":str(len(body)),
        })
        resp = json.loads(conn.getresponse().read().decode("utf-8"))
        conn.close()
        return resp
    except Exception as e:
        return {"error": str(e)}


# ── Public agent functions ───────────────────────────────────────────
def triage_shadow_regression(
    new_metrics:      dict,
    baseline_metrics: dict,
    n_users:          int = 1000,
) -> AgentResult:
    """
    Agent triage: compare new model vs baseline, identify regressions,
    summarise in natural language, recommend DEPLOY / HOLD / INVESTIGATE.
    Does NOT deploy. Human must review.
    """
    system = """You are a senior ML evaluation agent at a streaming company.
Your job is to triage shadow deployment results and recommend an action.
You NEVER deploy autonomously. You produce structured recommendations for human review.

CRITICAL CHECK: If ranker_auc is 0.50 or below, this indicates a broken or random ranker.
This is a RED ALARM — either features are weak, labels are wrong, or the problem setup is broken.
"It would be better in production" is NOT acceptable justification. Flag this explicitly.

Output valid JSON: {"action": "DEPLOY|HOLD|INVESTIGATE|ROLLBACK",
                    "justification": "...", "confidence": 0.0-1.0,
                    "requires_human_review": true,
                    "key_regressions": [], "key_improvements": [],
                    "red_alarms": []}"""

    user = f"""Shadow deployment results:
New model metrics:      {json.dumps(new_metrics, indent=2)}
Baseline metrics:       {json.dumps(baseline_metrics, indent=2)}
Users evaluated:        {n_users}

Identify regressions (>5% drop), improvements, and recommend action."""

    resp  = _call_openai_with_tools(system, user, AGENT_TOOLS[:1])
    try:
        content = resp["choices"][0]["message"].get("content","{}")
        parsed  = json.loads(content)
        return AgentResult(
            action=parsed.get("action","HOLD"),
            justification=parsed.get("justification","No justification"),
            confidence=float(parsed.get("confidence",0.5)),
            requires_human_review=True,   # ALWAYS true for deploy decisions
            raw_response=content,
        )
    except Exception:
        return AgentResult("HOLD","Agent response parse error",0.3,True)


def investigate_data_drift(
    drift_report: dict,
    recent_catalog_events: list[str] | None = None,
) -> AgentResult:
    """
    Agent investigation: given a drift report, identify likely causes
    and recommend remediation steps. Does NOT trigger retraining.
    """
    system = """You are a data quality agent. Given a drift report,
identify likely causes (new content wave, seasonal shift, schema change,
upstream pipeline failure, genuine user behaviour shift) and recommend
specific investigation steps. Output JSON:
{"action": "MONITOR|RETRAIN|INVESTIGATE_PIPELINE|HOLD_DEPLOYMENT",
 "likely_causes": [], "investigation_steps": [],
 "justification": "...", "confidence": 0.0-1.0,
 "requires_human_review": true}"""

    events_str = "\n".join(recent_catalog_events or ["No recent events available"])
    user = f"""Drift report:\n{json.dumps(drift_report, indent=2)}
Recent catalog/platform events:\n{events_str}"""

    resp = _call_openai_with_tools(system, user, AGENT_TOOLS[1:2])
    try:
        content = resp["choices"][0]["message"].get("content","{}")
        parsed  = json.loads(content)
        return AgentResult(
            action=parsed.get("action","INVESTIGATE_PIPELINE"),
            justification=parsed.get("justification",""),
            confidence=float(parsed.get("confidence",0.5)),
            requires_human_review=True,
            raw_response=content,
        )
    except Exception:
        return AgentResult("INVESTIGATE_PIPELINE","Parse error",0.3,True)


def policy_and_safety_gate(
    artwork_audits:   list[dict],
    explanation_samples: list[str],
    diversity_metrics: dict,
) -> AgentResult:
    """
    Policy compliance agent: checks artwork trust scores, explanation
    quality, diversity metrics, and safety signals before release.
    Blocks release if policy thresholds violated.
    """
    system = """You are a policy and safety agent for a streaming recommendation system.
Review artwork audits, explanation samples, and diversity metrics.
Flag any: misleading thumbnails (trust_score < 0.6), explanation hallucinations,
diversity failures (diversity_score < 0.4), or safety concerns.
Output JSON: {"action":"APPROVE|BLOCK|REVIEW",
              "policy_violations": [], "safety_concerns": [],
              "justification":"...", "confidence":0.0-1.0,
              "requires_human_review":true}"""

    low_trust = [a for a in artwork_audits if a.get("trust_score",1.0) < 0.6]
    user = f"""Artwork audits (n={len(artwork_audits)}, low-trust={len(low_trust)}):
{json.dumps(low_trust[:5], indent=2)}
Explanation samples: {json.dumps(explanation_samples[:3], indent=2)}
Diversity metrics: {json.dumps(diversity_metrics, indent=2)}"""

    resp = _call_openai_with_tools(system, user, AGENT_TOOLS[3:4])
    try:
        content = resp["choices"][0]["message"].get("content","{}")
        parsed  = json.loads(content)
        return AgentResult(
            action=parsed.get("action","REVIEW"),
            justification=parsed.get("justification",""),
            confidence=float(parsed.get("confidence",0.6)),
            requires_human_review=True,
            raw_response=content,
        )
    except Exception:
        return AgentResult("REVIEW","Parse error",0.3,True)


def generate_experiment_summary(
    experiment_name: str,
    metrics_before:  dict,
    metrics_after:   dict,
    config_changes:  list[str],
) -> str:
    """
    Generates a natural language experiment summary for engineering teams.
    Pure generation — no action, no deployment, just prose.
    """
    if not _OAI_KEY:
        return (f"Experiment '{experiment_name}': "
                f"NDCG changed from {metrics_before.get('ndcg_at_10','?')} "
                f"to {metrics_after.get('ndcg_at_10','?')}. "
                f"Changes: {', '.join(config_changes)}.")
    try:
        prompt = f"""Write a 3-paragraph engineering summary for this ML experiment.

Experiment: {experiment_name}
Config changes: {', '.join(config_changes)}
Before metrics: {json.dumps(metrics_before)}
After metrics: {json.dumps(metrics_after)}

Be specific about deltas. Flag any concerns. Use plain prose, no bullet points."""
        body = json.dumps({
            "model":"gpt-4o-mini",
            "messages":[{"role":"user","content":prompt}],
            "max_tokens":400,"temperature":0.3,
        }, ensure_ascii=True).encode("utf-8")
        ctx  = ssl.create_default_context()
        conn = http.client.HTTPSConnection("api.openai.com",timeout=10,context=ctx)
        conn.request("POST","/v1/chat/completions",body=body,headers={
            "Content-Type":"application/json","Authorization":f"Bearer {_OAI_KEY}",
            "Content-Length":str(len(body)),
        })
        resp = json.loads(conn.getresponse().read().decode("utf-8"))
        conn.close()
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Summary generation failed: {e}"
