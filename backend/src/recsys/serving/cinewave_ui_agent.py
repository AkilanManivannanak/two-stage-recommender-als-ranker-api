"""
cinewave_ui_agent.py
────────────────────
OmniParser-style UI agent for CineWave frontend evaluation.

Architecture (AutoGen-style multi-agent):
  OmniParser Agent  → parses screenshot into UI elements
  Planner Agent     → decides which UI action to take
  Actor Agent       → executes the action (click / type)
  Verifier Agent    → checks the outcome matches expectation
  Orchestrator      → coordinates the above agents

Usage:
  # Install dependencies
  pip install playwright pillow openai requests

  # Install browsers
  playwright install chromium

  # Run
  python3 cinewave_ui_agent.py --url http://localhost:3000

Add to: backend/src/recsys/serving/cinewave_ui_agent.py
"""

import argparse
import base64
import json
import re
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import Optional

import requests
from openai import OpenAI

# ── Try importing Playwright ──────────────────────────────────────────────
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

client = OpenAI()

# ── Data Classes ──────────────────────────────────────────────────────────

@dataclass
class UIElement:
    """A UI element parsed from a screenshot."""
    label: str
    element_type: str        # button / text / image / input / tab
    bounding_box: dict       # {x, y, width, height} in pixels
    confidence: float
    interactable: bool = True
    description: str = ""


@dataclass
class AgentAction:
    """An action an agent can take on the UI."""
    action_type: str         # click / type / scroll / screenshot / verify
    target_label: str
    value: Optional[str] = None
    reasoning: str = ""


@dataclass
class EvaluationResult:
    """Result of one UI agent evaluation run."""
    scenario: str
    steps: list = field(default_factory=list)
    ui_elements_found: list = field(default_factory=list)
    actions_taken: list = field(default_factory=list)
    verifications: list = field(default_factory=list)
    success: bool = False
    duration_ms: int = 0
    notes: str = ""


# ══════════════════════════════════════════════════════════════════════════
# AGENT 1 — OmniParser Agent
# Parses a screenshot into structured UI elements using GPT-4o Vision.
# Mirrors OmniParser's grounding of UI elements from pixels.
# ══════════════════════════════════════════════════════════════════════════

class OmniParserAgent:
    """
    Parses CineWave frontend screenshots into structured UI elements.

    Implements the core OmniParser capability:
      screenshot (pixels) → list of UIElement (label, type, bbox)

    Uses GPT-4o Vision as the vision-language backbone,
    following OmniParser's approach of using a VLM to identify
    and ground UI elements without requiring DOM access.
    """

    PARSE_PROMPT = """You are OmniParser — a UI element detection system.

Analyze this screenshot of a web application and identify ALL interactive
and informational UI elements.

For each element, provide:
1. label: descriptive name (e.g. "Voice microphone button", "Action Fan profile card")
2. type: button / text / image / input / tab / card / nav
3. x, y: approximate center position (0-100 as percentage of image)
4. width, height: approximate size (0-100 as percentage)
5. interactable: true/false
6. description: what this element does

Focus on:
- Profile selector cards (8 genre arms)
- Navigation buttons and tabs
- Voice/microphone button
- Movie poster cards
- ML Dashboard tabs
- Any metric displays or stat cards

Return ONLY valid JSON array:
[
  {
    "label": "...",
    "type": "...",
    "x": 50, "y": 30,
    "width": 20, "height": 10,
    "interactable": true,
    "description": "..."
  }
]"""

    def parse_screenshot(self, screenshot_b64: str) -> list[UIElement]:
        """Parse a base64 screenshot into UIElement list."""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": self.PARSE_PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{screenshot_b64}",
                                   "detail": "high"}}
                ]
            }],
            temperature=0.0,
            max_tokens=2000,
        )

        raw = response.choices[0].message.content.strip()

        # Extract JSON array
        try:
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                elements_data = json.loads(match.group())
            else:
                elements_data = json.loads(raw)
        except json.JSONDecodeError:
            return []

        elements = []
        for e in elements_data:
            elements.append(UIElement(
                label=e.get("label", "unknown"),
                element_type=e.get("type", "unknown"),
                bounding_box={
                    "x":      e.get("x", 50),
                    "y":      e.get("y", 50),
                    "width":  e.get("width", 10),
                    "height": e.get("height", 10),
                },
                confidence=e.get("confidence", 0.9),
                interactable=e.get("interactable", True),
                description=e.get("description", ""),
            ))

        return elements


# ══════════════════════════════════════════════════════════════════════════
# AGENT 2 — Planner Agent
# Decides which UI actions to take to complete a scenario.
# Implements ReAct-style reasoning: Thought → Action → Observation.
# ══════════════════════════════════════════════════════════════════════════

class PlannerAgent:
    """
    Plans UI actions to complete a given test scenario.

    Implements ReAct (Reasoning + Acting) pattern:
      Observation (UI elements) → Thought (what to do) → Action (how)

    This mirrors AutoGen's ReAct agent that reasons about
    available tools before selecting an action.
    """

    PLAN_PROMPT = """You are a UI test planner for CineWave, a movie recommendation system.

Available UI elements (from OmniParser):
{elements}

Your scenario: {scenario}

Plan the minimal sequence of actions to complete this scenario.
Think step by step (ReAct pattern):

Thought: What do I need to accomplish?
Available actions: click(label) | type(label, text) | scroll(direction) | verify(label, expected)

Return a JSON array of actions:
[
  {{
    "action_type": "click",
    "target_label": "exact label from elements list",
    "value": null,
    "reasoning": "why this action"
  }},
  {{
    "action_type": "verify",
    "target_label": "element to check",
    "value": "expected state or content",
    "reasoning": "what we expect to see"
  }}
]"""

    def plan_actions(
        self,
        elements: list[UIElement],
        scenario: str,
    ) -> list[AgentAction]:
        """Plan actions to complete a scenario given available UI elements."""

        elements_desc = "\n".join([
            f"- [{e.element_type}] '{e.label}': {e.description}"
            for e in elements if e.interactable
        ])

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": self.PLAN_PROMPT.format(
                    elements=elements_desc,
                    scenario=scenario,
                )
            }],
            temperature=0.0,
            max_tokens=1000,
        )

        raw = response.choices[0].message.content.strip()

        try:
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            actions_data = json.loads(match.group() if match else raw)
        except json.JSONDecodeError:
            return []

        return [
            AgentAction(
                action_type=a.get("action_type", "click"),
                target_label=a.get("target_label", ""),
                value=a.get("value"),
                reasoning=a.get("reasoning", ""),
            )
            for a in actions_data
        ]


# ══════════════════════════════════════════════════════════════════════════
# AGENT 3 — Actor Agent
# Executes UI actions using Playwright.
# ══════════════════════════════════════════════════════════════════════════

class ActorAgent:
    """
    Executes planned UI actions using Playwright browser automation.

    Translates abstract AgentAction objects (click, type, verify)
    into concrete browser interactions, using element labels
    from OmniParser to find elements via text matching.
    """

    def __init__(self, page):
        self.page = page

    def execute(self, action: AgentAction, elements: list[UIElement]) -> dict:
        """Execute one action. Returns {success, result, screenshot_b64}."""
        try:
            if action.action_type == "click":
                return self._click(action, elements)
            elif action.action_type == "type":
                return self._type(action, elements)
            elif action.action_type == "scroll":
                return self._scroll(action)
            elif action.action_type == "verify":
                return self._verify(action, elements)
            elif action.action_type == "screenshot":
                return self._screenshot()
            else:
                return {"success": False, "result": f"Unknown action: {action.action_type}"}
        except Exception as e:
            return {"success": False, "result": str(e)}

    def _click(self, action: AgentAction, elements: list[UIElement]) -> dict:
        """Click an element by label (text match or bounding box)."""
        label = action.target_label.lower()

        # Try text-based click first
        try:
            self.page.click(f"text=/{label}/i", timeout=3000)
            time.sleep(1)
            ss = self._take_screenshot()
            return {"success": True, "result": f"Clicked '{action.target_label}'",
                    "screenshot_b64": ss}
        except Exception:
            pass

        # Fallback: find element by bounding box from OmniParser
        for el in elements:
            if label in el.label.lower():
                bb = el.bounding_box
                viewport = self.page.viewport_size
                x = int(bb["x"] / 100 * viewport["width"])
                y = int(bb["y"] / 100 * viewport["height"])
                self.page.mouse.click(x, y)
                time.sleep(1)
                ss = self._take_screenshot()
                return {"success": True,
                        "result": f"Clicked at ({x},{y}) for '{action.target_label}'",
                        "screenshot_b64": ss}

        return {"success": False, "result": f"Could not find '{action.target_label}'"}

    def _type(self, action: AgentAction, elements: list[UIElement]) -> dict:
        """Type text into an input element."""
        try:
            self.page.fill(f"text=/{action.target_label}/i", action.value or "")
            return {"success": True, "result": f"Typed '{action.value}'"}
        except Exception as e:
            return {"success": False, "result": str(e)}

    def _scroll(self, action: AgentAction) -> dict:
        """Scroll the page."""
        direction = action.value or "down"
        delta = 300 if direction == "down" else -300
        self.page.mouse.wheel(0, delta)
        time.sleep(0.5)
        return {"success": True, "result": f"Scrolled {direction}"}

    def _verify(self, action: AgentAction, elements: list[UIElement]) -> dict:
        """Verify page state by re-parsing current screenshot."""
        ss = self._take_screenshot()
        return {
            "success": True,
            "result": f"Screenshot captured for verification of '{action.target_label}'",
            "screenshot_b64": ss,
            "needs_verification": True,
        }

    def _screenshot(self) -> dict:
        ss = self._take_screenshot()
        return {"success": True, "result": "Screenshot captured", "screenshot_b64": ss}

    def _take_screenshot(self) -> str:
        """Take screenshot and return as base64."""
        buf = self.page.screenshot(type="png")
        return base64.b64encode(buf).decode()


# ══════════════════════════════════════════════════════════════════════════
# AGENT 4 — Verifier Agent
# Checks outcomes using GPT-4o Vision.
# Mirrors MagenticOne's Critic Agent pattern.
# ══════════════════════════════════════════════════════════════════════════

class VerifierAgent:
    """
    Verifies action outcomes by comparing before/after screenshots.

    Implements MagenticOne's Critic Agent pattern:
    given an expected outcome and a screenshot, determine
    whether the outcome was achieved.
    """

    VERIFY_PROMPT = """You are a UI verification agent for CineWave.

Expected outcome: {expected}
Scenario: {scenario}

Look at this screenshot and determine:
1. Was the expected outcome achieved? (yes/no)
2. What do you observe on screen?
3. Any unexpected UI state?

Return JSON:
{{
  "outcome_achieved": true/false,
  "observations": "what you see",
  "confidence": 0.0-1.0,
  "issues": "any problems found or null"
}}"""

    def verify(
        self,
        screenshot_b64: str,
        expected: str,
        scenario: str,
    ) -> dict:
        """Verify screenshot matches expected outcome."""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": self.VERIFY_PROMPT.format(
                         expected=expected, scenario=scenario)},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{screenshot_b64}",
                                   "detail": "high"}}
                ]
            }],
            temperature=0.0,
            max_tokens=500,
        )

        raw = response.choices[0].message.content.strip()
        try:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            return json.loads(match.group() if match else raw)
        except json.JSONDecodeError:
            return {
                "outcome_achieved": False,
                "observations": raw,
                "confidence": 0.5,
                "issues": "JSON parse error",
            }


# ══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR — Coordinates all 4 agents
# Implements AutoGen GroupChat manager pattern.
# ══════════════════════════════════════════════════════════════════════════

class CineWaveUIOrchestrator:
    """
    Orchestrates the 4-agent UI evaluation pipeline.

    Implements AutoGen GroupChat Manager pattern:
    - Routes tasks between specialized agents
    - Maintains shared context (screenshots, elements, history)
    - Decides when to re-parse UI vs reuse cached elements
    - Produces structured evaluation report

    Test scenarios cover the full CineWave user journey:
    1. Profile selection → LinUCB arm selection
    2. Home feed verification → ALS recommendations visible
    3. Voice modal → wake-up system
    4. ML Dashboard → live metrics
    5. Diffusion page → DALL-E 3 poster generation
    """

    SCENARIOS = [
        {
            "name": "Profile Selection → LinUCB Arm",
            "description": "Select 'Sci-Fi Buff' profile and verify feed updates",
            "expected": "Home feed shows Sci-Fi/Fantasy/Thriller movies after profile selection",
            "url_path": "/",
        },
        {
            "name": "Home Feed → ALS Recommendations",
            "description": "Verify home feed shows movie posters with TMDB images",
            "expected": "At least 8 movie cards visible with poster images and titles",
            "url_path": "/",
        },
        {
            "name": "Voice Modal → Wake-Up System",
            "description": "Open voice modal and verify microphone interface appears",
            "expected": "Voice modal opens with microphone button and genre profile selector",
            "url_path": "/",
        },
        {
            "name": "ML Dashboard → Live Metrics",
            "description": "Navigate to ML dashboard and verify live API data loads",
            "expected": "ML Dashboard shows RL stats, GRU metrics, and OPE evaluation data",
            "url_path": "/ml",
        },
        {
            "name": "Diffusion Page → DDPM Stats",
            "description": "Navigate to diffusion page and verify DDPM schedule stats",
            "expected": "Diffusion page shows DDPM noise schedule bars and SNR values",
            "url_path": "/diffusion",
        },
    ]

    def __init__(self, base_url: str = "http://localhost:3000"):
        self.base_url = base_url
        self.omniparser  = OmniParserAgent()
        self.planner     = PlannerAgent()
        self.verifier    = VerifierAgent()
        self.results: list[EvaluationResult] = []

    def run_all_scenarios(self) -> list[EvaluationResult]:
        """Run all test scenarios. Returns structured evaluation results."""
        if not PLAYWRIGHT_AVAILABLE:
            print("[ERROR] Playwright not installed.")
            print("Run: pip install playwright && playwright install chromium")
            return self._run_mock_evaluation()

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            actor = ActorAgent(page)

            for scenario in self.SCENARIOS:
                result = self._run_scenario(page, actor, scenario)
                self.results.append(result)
                status = "✅ PASS" if result.success else "❌ FAIL"
                print(f"{status} {scenario['name']} ({result.duration_ms}ms)")

            browser.close()

        return self.results

    def _run_scenario(self, page, actor: ActorAgent, scenario: dict) -> EvaluationResult:
        """Run one scenario through the 4-agent pipeline."""
        t0 = time.perf_counter()
        result = EvaluationResult(scenario=scenario["name"])

        try:
            # Navigate to page
            url = self.base_url + scenario["url_path"]
            page.goto(url, wait_until="networkidle", timeout=15000)
            time.sleep(2)

            # Step 1: OmniParser Agent — parse UI
            result.steps.append("OmniParser: parsing screenshot")
            ss_b64 = base64.b64encode(page.screenshot(type="png")).decode()
            elements = self.omniparser.parse_screenshot(ss_b64)
            result.ui_elements_found = [e.label for e in elements]
            result.steps.append(f"OmniParser: found {len(elements)} UI elements")

            # Step 2: Planner Agent — plan actions
            result.steps.append("Planner: generating action plan")
            actions = self.planner.plan_actions(elements, scenario["description"])
            result.steps.append(f"Planner: planned {len(actions)} actions")

            # Step 3: Actor Agent — execute actions
            for action in actions[:4]:  # max 4 actions per scenario
                result.steps.append(f"Actor: {action.action_type} → {action.target_label}")
                action_result = actor.execute(action, elements)
                result.actions_taken.append({
                    "action": f"{action.action_type}({action.target_label})",
                    "success": action_result.get("success"),
                    "result":  action_result.get("result"),
                    "reasoning": action.reasoning,
                })

                # Step 4: Verifier Agent — check outcome after verify actions
                if action_result.get("needs_verification") and \
                   action_result.get("screenshot_b64"):
                    verification = self.verifier.verify(
                        action_result["screenshot_b64"],
                        scenario["expected"],
                        scenario["description"],
                    )
                    result.verifications.append(verification)
                    result.steps.append(
                        f"Verifier: achieved={verification.get('outcome_achieved')} "
                        f"conf={verification.get('confidence', 0):.2f}"
                    )

            # Final verification on current page state
            final_ss = base64.b64encode(page.screenshot(type="png")).decode()
            final_verification = self.verifier.verify(
                final_ss, scenario["expected"], scenario["description"]
            )
            result.verifications.append(final_verification)

            achieved = [v.get("outcome_achieved", False) for v in result.verifications]
            result.success = any(achieved)
            result.notes = final_verification.get("observations", "")

        except Exception as e:
            result.success = False
            result.notes = str(e)
            result.steps.append(f"ERROR: {e}")

        result.duration_ms = int((time.perf_counter() - t0) * 1000)
        return result

    def _run_mock_evaluation(self) -> list[EvaluationResult]:
        """
        Mock evaluation when Playwright is not available.
        Uses CineWave API endpoints to verify system state
        instead of browser UI interaction.
        """
        print("\n[MOCK MODE] Playwright not available.")
        print("Running API-based evaluation instead...\n")

        api_base = self.base_url.replace(":3000", ":8000")
        checks = [
            ("/healthz",              "System health",        lambda r: r.get("ok")),
            ("/ml/extensions/status", "ML extensions",        lambda r: r.get("sparse_training")),
            ("/rl/stats",             "REINFORCE policy",     lambda r: r.get("n_updates", 0) > 0),
            ("/eval/slice_ndcg",      "Slice NDCG",           lambda r: len(r.get("slices", {})) > 0),
            ("/drift",                "Drift monitor",        lambda r: r.get("status") == "HEALTHY"),
            ("/ml/ssl/summary",       "SSL GRU",              lambda r: r.get("available")),
            ("/diffusion/schedule",   "DDPM schedule",        lambda r: r.get("T") == 1000),
            ("/ab/experiments",       "A/B experiments",      lambda r: isinstance(r, list)),
        ]

        results = []
        for path, name, check_fn in checks:
            t0 = time.perf_counter()
            try:
                resp = requests.get(f"{api_base}{path}", timeout=5)
                data = resp.json()
                passed = check_fn(data)
            except Exception as e:
                passed = False
                data = {"error": str(e)}

            ms = int((time.perf_counter() - t0) * 1000)
            r = EvaluationResult(
                scenario=name,
                success=passed,
                duration_ms=ms,
                notes=str(data)[:120],
            )
            results.append(r)
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {status}  {name:<30} {ms:>5}ms")

        return results

    def print_report(self):
        """Print structured evaluation report."""
        print("\n" + "=" * 65)
        print("CINEWAVE UI AGENT — EVALUATION REPORT")
        print("=" * 65)

        passed = sum(1 for r in self.results if r.success)
        total  = len(self.results)
        print(f"\nOverall: {passed}/{total} scenarios passed\n")

        for r in self.results:
            status = "✅ PASS" if r.success else "❌ FAIL"
            print(f"{status}  {r.scenario} ({r.duration_ms}ms)")
            if r.ui_elements_found:
                print(f"      UI elements: {len(r.ui_elements_found)}")
            if r.verifications:
                for v in r.verifications:
                    conf = v.get("confidence", 0)
                    obs  = v.get("observations", "")[:80]
                    print(f"      Verifier [{conf:.0%}]: {obs}")
            if r.notes and not r.success:
                print(f"      Notes: {r.notes[:100]}")
            print()

        print("=" * 65)
        print("Agent pipeline: OmniParser → Planner → Actor → Verifier")
        print("Framework: AutoGen-style orchestration + MagenticOne Critic")
        print("Vision backbone: GPT-4o (OmniParser grounding)")
        print("=" * 65)

    def save_report(self, path: str = "ui_agent_report.json"):
        """Save full report as JSON."""
        report = {
            "summary": {
                "total":  len(self.results),
                "passed": sum(1 for r in self.results if r.success),
                "agents": ["OmniParserAgent", "PlannerAgent",
                           "ActorAgent", "VerifierAgent"],
                "framework": "AutoGen-style orchestration",
            },
            "scenarios": [
                {
                    "scenario":    r.scenario,
                    "success":     r.success,
                    "duration_ms": r.duration_ms,
                    "steps":       r.steps,
                    "ui_elements": r.ui_elements_found,
                    "actions":     r.actions_taken,
                    "verifications": r.verifications,
                    "notes":       r.notes,
                }
                for r in self.results
            ],
        }
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {path}")


# ── Entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CineWave OmniParser UI Agent"
    )
    parser.add_argument(
        "--url", default="http://localhost:3000",
        help="CineWave frontend URL"
    )
    parser.add_argument(
        "--scenario", type=int, default=None,
        help="Run single scenario by index (0-4)"
    )
    parser.add_argument(
        "--report", default="ui_agent_report.json",
        help="Output report path"
    )
    args = parser.parse_args()

    print("CineWave UI Agent")
    print("─────────────────────────────────────────────────────────")
    print(f"Frontend URL:  {args.url}")
    print(f"Playwright:    {'available' if PLAYWRIGHT_AVAILABLE else 'NOT installed (mock mode)'}")
    print(f"Agents:        OmniParser · Planner · Actor · Verifier")
    print(f"Framework:     AutoGen-style multi-agent orchestration")
    print("─────────────────────────────────────────────────────────\n")

    orchestrator = CineWaveUIOrchestrator(base_url=args.url)

    if args.scenario is not None:
        orchestrator.SCENARIOS = [orchestrator.SCENARIOS[args.scenario]]

    orchestrator.run_all_scenarios()
    orchestrator.print_report()
    orchestrator.save_report(args.report)


if __name__ == "__main__":
    main()
