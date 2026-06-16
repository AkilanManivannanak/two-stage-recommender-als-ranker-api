"""
chain_of_thought_intent.py
─────────────────────────
Chain-of-thought intent extraction for CineWave voice pipeline.

Compares three prompting strategies:
  A) Direct prompting
  B) Chain-of-thought (CoT)
  C) Few-shot + CoT

Drop-in replacement for GPT-4o intent extraction in smart_explain.py.
Add to: backend/src/recsys/serving/chain_of_thought_intent.py
"""

import json
import time
from typing import Literal
from openai import OpenAI

client = OpenAI()

# ── Strategy A — Direct prompting ────────────────────────────────────────────
DIRECT_PROMPT = """Extract movie recommendation intent from the user's spoken query.

Return JSON with keys: genres (list), similar_to (str or null), year_min (int or null), mood (str or null).

Query: {query}

JSON:"""

# ── Strategy B — Chain-of-Thought ────────────────────────────────────────────
COT_PROMPT = """You are a movie recommendation intent extractor. Think step by step.

User query: "{query}"

Step 1 — What emotional tone or mood does the user want?
Step 2 — Which of these genres match that mood?
         [Action, Comedy, Drama, Horror, Sci-Fi, Romance, Thriller, Documentary, Crime, Fantasy]
Step 3 — Did the user mention a specific movie title as reference? (similar_to)
Step 4 — Did the user indicate a time period preference? (year_min)
Step 5 — Summarize the final structured intent.

Think through each step, then output ONLY valid JSON:
{{"genres": [...], "similar_to": null or "Title", "year_min": null or 1970, "mood": "..."}}

Reasoning:"""

# ── Strategy C — Few-shot + CoT ───────────────────────────────────────────────
FEW_SHOT_COT_PROMPT = """You are a movie recommendation intent extractor. Think step by step.

EXAMPLE 1:
Query: "Something like Inception but scarier"
Reasoning:
  Step 1: User wants psychological fear, dark atmosphere
  Step 2: Genres: Thriller, Horror, Sci-Fi (mind-bending)
  Step 3: similar_to = "Inception"
  Step 4: No year mentioned → year_min = 1970 (default)
  Step 5: Intent is psychological thriller with sci-fi elements
Output: {{"genres": ["Thriller", "Horror", "Sci-Fi"], "similar_to": "Inception", "year_min": 1970, "mood": "psychological fear"}}

EXAMPLE 2:
Query: "I want something funny for date night"
Reasoning:
  Step 1: User wants light, fun, romantic mood
  Step 2: Genres: Comedy, Romance
  Step 3: No reference title
  Step 4: No year mentioned
  Step 5: Light romantic comedy
Output: {{"genres": ["Comedy", "Romance"], "similar_to": null, "year_min": null, "mood": "light and romantic"}}

EXAMPLE 3:
Query: "Old war movies like Saving Private Ryan"
Reasoning:
  Step 1: User wants serious, realistic war drama
  Step 2: Genres: Drama, Action (war subgenre)
  Step 3: similar_to = "Saving Private Ryan"
  Step 4: "Old" suggests classic era → year_min = 1960
  Step 5: Serious war drama, classic era
Output: {{"genres": ["Drama", "Action"], "similar_to": "Saving Private Ryan", "year_min": 1960, "mood": "serious war drama"}}

NOW YOUR TURN:
Query: "{query}"
Reasoning:"""


def extract_intent(
    query: str,
    strategy: Literal["direct", "cot", "few_shot_cot"] = "cot",
    model: str = "gpt-4o",
) -> dict:
    """
    Extract structured movie intent from a voice query.

    Args:
        query: Raw transcript from Whisper STT
        strategy: Prompting strategy to use
        model: OpenAI model

    Returns:
        dict with keys: genres, similar_to, year_min, mood,
                        strategy_used, reasoning (if CoT), latency_ms
    """
    prompts = {
        "direct":       DIRECT_PROMPT.format(query=query),
        "cot":          COT_PROMPT.format(query=query),
        "few_shot_cot": FEW_SHOT_COT_PROMPT.format(query=query),
    }

    prompt = prompts[strategy]
    t0 = time.perf_counter()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=400,
    )

    latency_ms = round((time.perf_counter() - t0) * 1000)
    raw = response.choices[0].message.content.strip()

    # Parse JSON from response
    try:
        # Extract JSON block if CoT included reasoning text
        if "{" in raw:
            json_start = raw.rfind("{")
            json_end   = raw.rfind("}") + 1
            parsed = json.loads(raw[json_start:json_end])
        else:
            parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"genres": [], "similar_to": None, "year_min": None, "mood": None}

    return {
        **parsed,
        "strategy_used": strategy,
        "reasoning":     raw if strategy != "direct" else None,
        "latency_ms":    latency_ms,
    }


# ── Evaluation harness ────────────────────────────────────────────────────────
TEST_QUERIES = [
    # (query, expected_genres, expected_similar_to)
    ("Show me something like Inception", ["Sci-Fi", "Thriller"], "Inception"),
    ("I want a scary movie", ["Horror", "Thriller"], None),
    ("Funny romantic movie for tonight", ["Comedy", "Romance"], None),
    ("Something with Tom Hanks", ["Drama"], None),
    ("Old war movies", ["Drama", "Action"], None),
    ("Mind-bending sci-fi", ["Sci-Fi", "Thriller"], None),
    ("Show me a documentary about nature", ["Documentary"], None),
    ("Action movies from the 80s", ["Action"], None),
    ("Something dark like Black Mirror", ["Sci-Fi", "Thriller"], "Black Mirror"),
    ("Feel-good family movie", ["Comedy", "Drama"], None),
]


def evaluate_strategies(n_queries: int = 10) -> dict:
    """
    Compare direct vs CoT vs few-shot CoT on genre extraction accuracy.
    Returns accuracy and latency for each strategy.
    """
    results = {s: {"correct": 0, "total": 0, "latency_ms": []}
               for s in ["direct", "cot", "few_shot_cot"]}

    for query, expected_genres, _ in TEST_QUERIES[:n_queries]:
        for strategy in ["direct", "cot", "few_shot_cot"]:
            result = extract_intent(query, strategy)
            predicted = set(g.lower() for g in result.get("genres", []))
            expected  = set(g.lower() for g in expected_genres)

            # Partial match: at least 1 expected genre found
            correct = len(predicted & expected) > 0
            results[strategy]["correct"] += int(correct)
            results[strategy]["total"]   += 1
            results[strategy]["latency_ms"].append(result["latency_ms"])

    # Compute summary
    summary = {}
    for strategy, r in results.items():
        acc = r["correct"] / r["total"] if r["total"] > 0 else 0
        avg_lat = sum(r["latency_ms"]) / len(r["latency_ms"]) if r["latency_ms"] else 0
        summary[strategy] = {
            "accuracy":     round(acc, 3),
            "avg_latency_ms": round(avg_lat),
            "correct":      r["correct"],
            "total":        r["total"],
        }

    return summary


if __name__ == "__main__":
    print("CineWave CoT Intent Extractor — Strategy Comparison")
    print("=" * 55)

    # Single query demo
    q = "Show me something like Inception but darker and scarier"
    print(f"\nQuery: '{q}'\n")

    for strategy in ["direct", "cot", "few_shot_cot"]:
        result = extract_intent(q, strategy)
        print(f"[{strategy.upper()}]")
        print(f"  Genres:     {result.get('genres')}")
        print(f"  Similar to: {result.get('similar_to')}")
        print(f"  Mood:       {result.get('mood')}")
        print(f"  Latency:    {result['latency_ms']}ms")
        if result.get("reasoning"):
            print(f"  Reasoning:  {result['reasoning'][:120]}...")
        print()

    # Evaluation
    print("\nRunning strategy evaluation on 10 test queries...")
    summary = evaluate_strategies(10)
    print("\nRESULTS:")
    print(f"{'Strategy':<20} {'Accuracy':>10} {'Avg Latency':>14} {'Correct/Total':>14}")
    print("-" * 60)
    for strategy, r in summary.items():
        print(f"{strategy:<20} {r['accuracy']:>10.1%} {r['avg_latency_ms']:>12}ms {r['correct']:>6}/{r['total']:<6}")
