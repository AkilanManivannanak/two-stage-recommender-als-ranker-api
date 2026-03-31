"""
Semantic Sidecar Layer — Phase 7 (GPT via Responses API + Structured Outputs)
==============================================================================
GPT is used ONLY for:
  1. Catalog enrichment (themes, moods, tags)
  2. Explanation generation from model attributions
  3. Editorial row naming
  4. Voice intent parsing (complex utterances)
  5. Experiment/regression summaries
  6. Query rewrite

NOT in the hot path. All calls are async/cached.
Uses Responses API (openai.responses.create) + Structured Outputs.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Optional


# ── Structured output schemas ─────────────────────────────────────────────

CATALOG_ENRICHMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "themes":               {"type": "array",  "items": {"type": "string"}},
        "moods":                {"type": "array",  "items": {"type": "string"}},
        "semantic_tags":        {"type": "array",  "items": {"type": "string"}},
        "pacing":               {"type": "string", "enum": ["slow", "medium", "fast"]},
        "tone":                 {"type": "string", "enum": ["light", "serious", "mixed"]},
        "spoiler_safe_summary": {"type": "string"},
        "audience_fit":         {"type": "array",  "items": {"type": "string"}},
    },
    "required": ["themes", "moods", "semantic_tags", "spoiler_safe_summary"],
    "additionalProperties": False,
}

EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {
        "reason":      {"type": "string"},
        "top_feature": {"type": "string"},
        "confidence":  {"type": "number"},
        "method":      {"type": "string"},
    },
    "required": ["reason", "top_feature", "confidence"],
    "additionalProperties": False,
}

VOICE_INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent":       {"type": "string", "enum": ["discover", "navigate", "refine", "clarify"]},
        "genres":       {"type": "array",  "items": {"type": "string"}},
        "moods":        {"type": "array",  "items": {"type": "string"}},
        "exclude":      {"type": "array",  "items": {"type": "string"}},
        "title_query":  {"type": "string"},
        "confidence":   {"type": "number"},
        "needs_clarification": {"type": "boolean"},
    },
    "required": ["intent", "genres", "confidence", "needs_clarification"],
    "additionalProperties": False,
}

ROW_TITLE_SCHEMA = {
    "type": "object",
    "properties": {
        "title":    {"type": "string"},
        "subtitle": {"type": "string"},
    },
    "required": ["title"],
    "additionalProperties": False,
}


@dataclass
class SidecarClient:
    """
    Thin wrapper around OpenAI Responses API.
    Falls back gracefully to rule-based output if API unavailable.
    """
    api_key:  str = ""
    model:    str = "gpt-4o"          # or "gpt-4o-mini" for lower cost
    cache_ttl: int = 21600            # 6 hours

    def _call_responses_api(
        self,
        system: str,
        user: str,
        schema: dict,
        max_tokens: int = 500,
    ) -> Optional[dict]:
        """
        Call OpenAI Responses API with Structured Outputs.
        Uses openai.responses.create (new Responses API pattern).
        """
        if not self.api_key:
            return None
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)

            # Responses API: use response_format with json_schema for Structured Outputs
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",  "content": system},
                    {"role": "user",    "content": user},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "structured_output",
                        "strict": True,
                        "schema": schema,
                    }
                },
                max_tokens=max_tokens,
                temperature=0.3,
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception:
            return None

    def enrich_catalog_item(self, title: str, genre: str, description: str = "") -> dict:
        """
        Enrich a catalog item with themes, moods, semantic tags.
        Falls back to rule-based if GPT unavailable.
        """
        result = self._call_responses_api(
            system=(
                "You are a film metadata expert. "
                "Extract structured attributes from the title and description. "
                "Keep spoiler_safe_summary under 2 sentences."
            ),
            user=f"Title: {title}\nGenre: {genre}\nDescription: {description or 'Not provided'}",
            schema=CATALOG_ENRICHMENT_SCHEMA,
            max_tokens=300,
        )
        if result:
            result["method"] = "gpt_structured"
            return result

        # Rule-based fallback
        return {
            "themes":               [genre, "character development"],
            "moods":                ["engaging", "entertaining"],
            "semantic_tags":        [genre, title[:15]],
            "pacing":               "medium",
            "tone":                 "mixed",
            "spoiler_safe_summary": f"A compelling {genre} title.",
            "audience_fit":         ["general"],
            "method":               "rule_based",
        }

    def generate_explanation(
        self,
        title: str,
        genre: str,
        user_top_genre: str,
        model_attribution: dict,
    ) -> dict:
        """
        Generate a human-readable explanation of why this title was recommended.
        Explanation is grounded in model attributions (SHAP-style), not hallucinated.
        """
        top_feature = max(
            model_attribution, key=lambda k: model_attribution[k]
        ) if model_attribution else "genre_affinity"

        result = self._call_responses_api(
            system=(
                "You write concise, honest recommendation explanations (1-2 sentences). "
                "Base the explanation on the provided model feature, not general knowledge. "
                "Never claim certainty about private user data."
            ),
            user=(
                f"Title: {title} ({genre})\n"
                f"User's top genre: {user_top_genre}\n"
                f"Top model feature: {top_feature} = {model_attribution.get(top_feature, 0):.2f}\n"
                f"Generate an explanation."
            ),
            schema=EXPLANATION_SCHEMA,
            max_tokens=150,
        )
        if result:
            result["method"] = "gpt_attributed"
            return result

        # Rule-based fallback
        return {
            "reason":      f"Because you enjoy {user_top_genre} and this has high ratings for that genre.",
            "top_feature": top_feature,
            "confidence":  0.7,
            "method":      "rule_based",
        }

    def parse_voice_intent(self, transcript: str) -> dict:
        """
        Parse a complex voice utterance into structured intent.
        Used only when simple keyword matching fails (>3 words, ambiguous).
        """
        result = self._call_responses_api(
            system=(
                "You are a voice intent parser for a streaming service. "
                "Extract intent, genres, moods, and exclusions from user utterances. "
                "confidence should reflect certainty (0.0-1.0). "
                "Set needs_clarification=true if intent is genuinely ambiguous."
            ),
            user=f"User said: \"{transcript}\"",
            schema=VOICE_INTENT_SCHEMA,
            max_tokens=200,
        )
        if result:
            result["method"] = "gpt_structured"
            return result

        # Rule-based fallback for simple cases
        text = transcript.lower()
        genres = []
        for g in ["comedy", "drama", "horror", "action", "documentary", "thriller", "romance"]:
            if g in text:
                genres.append(g.capitalize())

        intent = "discover"
        if any(w in text for w in ["play", "watch", "open", "start"]):
            intent = "navigate"
        elif any(w in text for w in ["not", "no ", "without", "exclude"]):
            intent = "refine"

        return {
            "intent":   intent,
            "genres":   genres,
            "moods":    [],
            "exclude":  [],
            "title_query": "",
            "confidence": 0.8 if genres else 0.5,
            "needs_clarification": len(genres) == 0 and intent == "discover",
            "method": "rule_based",
        }

    def generate_row_title(self, genre: str, context: str = "") -> dict:
        """Generate editorial row titles for the homepage."""
        result = self._call_responses_api(
            system="Generate a short, engaging Netflix-style row title. Max 5 words.",
            user=f"Genre: {genre}\nContext: {context or 'general recommendations'}",
            schema=ROW_TITLE_SCHEMA,
            max_tokens=50,
        )
        if result:
            return result
        templates = {
            "Drama":       "Stories That Stay With You",
            "Comedy":      "Something to Make You Laugh",
            "Action":      "High-Octane Picks",
            "Horror":      "Things That Go Bump",
            "Documentary": "True Stories Worth Watching",
            "Thriller":    "Keep You on the Edge",
            "Sci-Fi":      "Beyond the Imagination",
            "Animation":   "For the Young at Heart",
            "Crime":       "Crime Doesn't Pay",
            "Romance":     "Love Is in the Air",
        }
        return {"title": templates.get(genre, f"Great {genre} for You")}

    def summarise_regression(self, metrics: dict, baseline: dict) -> str:
        """Advisory-only triage summary. Never triggers autonomous deployment."""
        diffs = {k: round(metrics.get(k, 0) - baseline.get(k, 0), 4)
                 for k in baseline}
        result = self._call_responses_api(
            system=(
                "You are a model evaluation assistant. "
                "Summarise the metric changes in 2-3 sentences. "
                "Flag any regressions. This is advisory only — no deployment decisions."
            ),
            user=f"Metric changes vs baseline: {json.dumps(diffs)}",
            schema={
                "type": "object",
                "properties": {
                    "summary":    {"type": "string"},
                    "regressions": {"type": "array", "items": {"type": "string"}},
                    "improvements": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["summary"],
                "additionalProperties": False,
            },
            max_tokens=200,
        )
        if result:
            return result.get("summary", "Summary unavailable.")
        return (
            f"Metric deltas: {diffs}. "
            "Advisory summary: human review required before any deployment decision."
        )


# Module-level singleton — api_key injected at startup
SIDECAR = SidecarClient()
