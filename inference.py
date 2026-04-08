from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Safe-load dotenv: Works locally, skips in environments without python-dotenv.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests
from openai import OpenAI

# ─────────────────────────────────────────────
# Configuration — all from environment variables
# ─────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
HF_TOKEN     = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is missing.")

ENV_BASE_URL = "http://localhost:8000"

TASKS = ["flood_easy", "earthquake_medium", "compound_hard"]
SEED  = 42

client = OpenAI(
    api_key  = HF_TOKEN,
    base_url = API_BASE_URL,
)

# ─────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI disaster relief coordinator.
You will receive the current state of a disaster simulation as JSON.
Your job is to choose the single best action to take right now.

You MUST respond with a single valid JSON object — nothing else.
No markdown, no explanation, no code blocks. Raw JSON only.

Available actions (pick exactly one):

1.  Assess a zone to gather information:
    {"type": "AssessZone", "zone_id": <int>}

2.  Dispatch resources to a zone:
    {"type": "DispatchResource", "resource_type": "<type>", "quantity": <int>, "zone_id": <int>}
    resource_type options: "ambulances", "food_trucks", "field_hospitals", "rescue_teams"

3.  Establish a field hospital in a zone (consumes 1 field_hospital resource):
    {"type": "EstablishFieldHospital", "zone_id": <int>}

4.  Open a blocked or degraded road in a zone:
    {"type": "OpenAlternativeRoute", "zone_id": <int>}

5.  Request external aid (compound_hard: MUST be called before hour 12):
    {"type": "RequestExternalAid", "resource_type": "<type>"}

6.  Set a zone priority:
    {"type": "PrioritizeZone", "zone_id": <int>, "priority": "<CRITICAL|HIGH|MEDIUM|LOW>"}

7.  Coordinate with an NGO:
    {"type": "CoordinateWithNGO", "ngo_id": "<string>", "task": "<string>"}

Strategy tips:
- Prioritise CRITICAL zones with CLEAR roads first.
- Do NOT dispatch to BLOCKED roads — you will be penalised.
- Use OpenAlternativeRoute to unblock roads before dispatching.
- compound_hard has many zones — prioritize DispatchResource over OpenAlternativeRoute.
- Only use OpenAlternativeRoute if a zone is BLOCKED AND has high priority needs.
- Do NOT repeatedly open routes — each zone only needs it once.
- Match resource type to zone needs:
    ambulances      → MEDICAL needs
    food_trucks     → FOOD needs
    field_hospitals → SHELTER or MEDICAL needs
    rescue_teams    → RESCUE needs
- Request external aid early in compound_hard (before hour 12).
- Avoid assessing the same zone twice — you will be penalised.
"""

CORRECTION_PROMPT = """Your previous response was not valid JSON or was not a recognised action.
You MUST respond with ONLY a raw JSON object with a "type" field.
Example: {"type": "AssessZone", "zone_id": 0}
No markdown, no explanation — raw JSON only."""

def get_fallback_action(obs: dict) -> dict:
    resources = obs.get("resources", {})
    need_resource_map = {
        "MEDICAL": "ambulances",
        "RESCUE": "rescue_teams",
        "FOOD": "food_trucks",
        "SHELTER": "field_hospitals",
    }
    for z in obs.get("zones", []):
        if not z.get("served") and z.get("road_status") != "BLOCKED":
            for need in z.get("needs", []):
                resource = need_resource_map.get(need)
                if resource and resources.get(resource, 0) > 0:
                    return {
                        "type": "DispatchResource",
                        "resource_type": resource,
                        "quantity": 2,
                        "zone_id": z["zone_id"],
                    }
    return {"type": "AssessZone", "zone_id": 0}


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={'true' if done else 'false'} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    # Matching the visual format of their example exactly
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}", flush=True)


# ─────────────────────────────────────────────
# Environment API helpers
# ─────────────────────────────────────────────

def env_reset(task_id: str, seed: int) -> Dict[str, Any]:
    resp = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id, "seed": seed},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(
        f"{ENV_BASE_URL}/step",
        json=action,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────
# LLM action selection
# ─────────────────────────────────────────────

def observation_to_prompt(obs: Dict[str, Any]) -> str:
    zones     = obs.get("zones", [])
    resources = obs.get("resources", {})
    hour      = obs.get("simulation_hour", 0)
    weather   = obs.get("weather", {})
    events    = obs.get("active_events", [])
    baseline  = obs.get("baseline_survival_rate", 0)
    current   = obs.get("current_survival_rate", 0)

    critical_zones = [z for z in zones if z.get("damage_level") == "CRITICAL" and not z.get("served")]
    blocked_zones  = [z for z in zones if z.get("road_status") == "BLOCKED"]
    unserved_zones = [z for z in zones if not z.get("served")]

    lines = [
        f"SIMULATION HOUR: {hour}",
        f"SURVIVAL RATE: current={current:.3f}  baseline={baseline:.3f}  delta={current - baseline:+.3f}",
        f"WEATHER: {weather.get('condition', 'CLEAR')}",
        "",
        "AVAILABLE RESOURCES:",
        f"  ambulances={resources.get('ambulances', 0)}  "
        f"food_trucks={resources.get('food_trucks', 0)}  "
        f"field_hospitals={resources.get('field_hospitals', 0)}  "
        f"rescue_teams={resources.get('rescue_teams', 0)}",
        "",
        f"CRITICAL UNSERVED ZONES ({len(critical_zones)}):",
    ]
    max_zones = 4 if len(zones) > 15 else 8
    for z in critical_zones[:max_zones]:
        lines.append(
            f"  zone_id={z['zone_id']}  pop={z['population']}  "
            f"road={z['road_status']}  needs={z['needs']}  "
            f"casualties={z['casualties']}"
        )

    lines += [
        "",
        f"BLOCKED ROAD ZONES ({len(blocked_zones)}): "
        + ", ".join(str(z["zone_id"]) for z in blocked_zones[:5]),
        "",
        f"TOTAL UNSERVED ZONES: {len(unserved_zones)}",
    ]

    if events:
        lines += ["", "RECENT EVENTS:"]
        for e in events[-3:]:
            lines.append(f"  {e}")

    return "\n".join(lines)


def parse_action_from_llm(raw: str) -> Optional[Dict[str, Any]]:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        inner = [l for l in lines if not l.startswith("```")]
        raw = "\n".join(inner).strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "type" in obj:
            return obj
        return None
    except json.JSONDecodeError:
        start = raw.find("{")
        end   = raw.rfind("}")
        if start != -1 and end != -1:
            try:
                obj = json.loads(raw[start:end + 1])
                if isinstance(obj, dict) and "type" in obj:
                    return obj
            except json.JSONDecodeError:
                pass
        return None


def choose_action(obs: Dict[str, Any], conversation: list) -> Dict[str, Any]:
    # Keep conversation short — only last 6 messages + system prompt
    if len(conversation) > 7:
        conversation[1:] = conversation[-6:]

    user_msg = observation_to_prompt(obs)
    conversation.append({"role": "user", "content": user_msg})

    raw = ""

    # First attempt
    try:
        response = client.chat.completions.create(
            model       = MODEL_NAME,
            messages    = conversation,
            temperature = 0.2,
            max_tokens  = 512,
            stream      = False,
        )
        raw    = response.choices[0].message.content or ""
        action = parse_action_from_llm(raw)
        if action is not None:
            conversation.append({"role": "assistant", "content": raw})
            return action
    except Exception as e:
        # print(f"  [LLM ERROR] First attempt failed: {e}", file=sys.stderr)
        pass

    # Retry with correction prompt
    conversation.append({"role": "assistant", "content": raw})
    conversation.append({"role": "user",      "content": CORRECTION_PROMPT})
    try:
        response = client.chat.completions.create(
            model       = MODEL_NAME,
            messages    = conversation,
            temperature = 0.0,
            max_tokens  = 256,
            stream      = False,
        )
        raw    = response.choices[0].message.content or ""
        action = parse_action_from_llm(raw)
        if action is not None:
            conversation.append({"role": "assistant", "content": raw})
            return action
    except Exception as e:
        # print(f"  [LLM ERROR] Retry attempt failed: {e}", file=sys.stderr)
        pass

    # Fallback
    fallback = get_fallback_action(obs)
    # print("  [FALLBACK] Using smart dispatch", file=sys.stderr)
    conversation.append({"role": "assistant", "content": json.dumps(fallback)})
    return fallback


# ─────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────

def run_task(task_id: str) -> float:
    log_start(task=task_id, env="disaster-relief-openenv", model=MODEL_NAME)

    rewards_history = []
    conversation    = [{"role": "system", "content": SYSTEM_PROMPT}]
    final_score     = 0.0
    step_num        = 0
    cumulative      = 0.0

    try:
        obs = env_reset(task_id, SEED)
    except Exception:
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return 0.0

    while True:
        step_num += 1
        error_msg = None

        try:
            action = choose_action(obs, conversation)
        except Exception as e:
            action = get_fallback_action(obs)
            error_msg = str(e)[:50]

        action_type = action.get("type", "Unknown")

        try:
            result     = env_step(action)
            reward     = result.get("reward", 0.0)
            cumulative = result.get("cumulative_reward", cumulative)
            done       = result.get("done", False)
            obs        = result.get("observation", obs)
        except Exception as e:
            log_step(step=step_num, action=action_type, reward=0.0, done=False, error=str(e)[:50])
            log_end(success=False, steps=step_num, score=cumulative, rewards=rewards_history)
            return cumulative

        rewards_history.append(reward)
        log_step(step=step_num, action=action_type, reward=reward, done=done, error=error_msg)

        if done:
            final_score = max(0.001, min(cumulative, 0.994))
            break

    success = final_score >= 0.10
    log_end(success=success, steps=step_num, score=final_score, rewards=rewards_history)
    return final_score


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────

def main() -> None:
    start_time = time.time()
    scores     = {}

    for task_id in TASKS:
        task_start = time.time()
        score      = run_task(task_id)
        scores[task_id] = score
        elapsed = time.time() - task_start
        # print(f"  [TIMING] task={task_id} elapsed={elapsed:.1f}s", file=sys.stderr)

        total_elapsed = time.time() - start_time
        if total_elapsed > 1100:
            # print(
            #     "[WARNING] Approaching 20-minute limit — stopping early.",
            #     file=sys.stderr,
            # )
            break

    # print("\n[SUMMARY]", file=sys.stderr)
    # for task_id, score in scores.items():
    #     print(f"  {task_id}: {score:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()