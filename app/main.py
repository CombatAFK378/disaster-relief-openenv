from __future__ import annotations

from fastapi import Body, FastAPI, HTTPException, Request
from pydantic import TypeAdapter

from app.environment import DisasterEnv
from app.models import (
    Action,
    EpisodeState,
    HealthResponse,
    Observation,
    ResetRequest,
    StepResult,
)

app = FastAPI(
    title="Disaster Relief Coordinator — OpenEnv",
    description=(
        "## Disaster Relief Coordinator\n\n"
        "An **OpenEnv-compliant** reinforcement learning environment where AI agents "
        "learn to coordinate disaster relief resources across a fully simulated "
        "disaster scenario — trained to save more lives than human coordinators "
        "operating under panic and time pressure.\n\n"
        "Built for the **Meta PyTorch x Hugging Face OpenEnv AI Hackathon 2026**. "
        "Directly aligned with Meta's **AI for Good** humanitarian response program "
        "and the Texas A&M UrbanResilience.AI partnership.\n\n"
        "---\n\n"
        "### How to use\n\n"
        "**1. Start an episode**\n\n"
        "    POST /reset  →  { \"task_id\": \"flood_easy\", \"seed\": 42 }\n\n"
        "**2. Take actions — repeat up to 12 steps (72 simulated hours)**\n\n"
        "    POST /step  →  { \"type\": \"AssessZone\", \"zone_id\": 2 }\n"
        "    POST /step  →  { \"type\": \"DispatchResource\", \"resource_type\": \"ambulances\", \"quantity\": 2, \"zone_id\": 3 }\n"
        "    POST /step  →  { \"type\": \"EstablishFieldHospital\", \"zone_id\": 4 }\n"
        "    POST /step  →  { \"type\": \"OpenAlternativeRoute\", \"zone_id\": 1 }\n"
        "    POST /step  →  { \"type\": \"RequestExternalAid\", \"resource_type\": \"food_trucks\" }\n"
        "    POST /step  →  { \"type\": \"PrioritizeZone\", \"zone_id\": 0, \"priority\": \"CRITICAL\" }\n"
        "    POST /step  →  { \"type\": \"CoordinateWithNGO\", \"ngo_id\": \"UNHCR-01\", \"task\": \"shelter_setup\" }\n\n"
        "**3. Inspect full state at any time (read-only)**\n\n"
        "    GET /state\n\n"
        "---\n\n"
        "### Tasks\n\n"
        "| Task ID | Difficulty | Zones | Roads Blocked | Expected Score |\n"
        "|---|---|---|---|---|\n"
        "| flood_easy | Easy | 8 | 10% | 0.65-0.75 |\n"
        "| earthquake_medium | Medium | 15 | 40% | 0.40-0.55 |\n"
        "| compound_hard | Hard | 20 | 60% | 0.15-0.28 |\n\n"
        "---\n\n"
        "### Reward breakdown (fires every step)\n\n"
        "| Component | Value |\n"
        "|---|---|\n"
        "| People reached this step | +0.02 x people (max 500) |\n"
        "| CRITICAL zone cleared within 24 h | +0.10 per zone |\n"
        "| Hospital capacity never exceeded | +0.15 terminal bonus |\n"
        "| Survival improvement over baseline | +0.20 x delta terminal |\n"
        "| Resource sent to BLOCKED zone | -0.10 |\n"
        "| CRITICAL zone unserved > 12 h | -0.20 (once per zone) |\n"
        "| Duplicate AssessZone | -0.05 |\n"
    ),
    version="1.0.0",
)

# ─────────────────────────────────────────────
# Global singleton — one active episode at a time
# ─────────────────────────────────────────────

env = DisasterEnv()

_action_adapter: TypeAdapter[Action] = TypeAdapter(Action)


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Liveness probe",
)
def health() -> HealthResponse:
    """
    Returns {"status": "ok"} immediately.
    Used by Hugging Face Spaces as a liveness ping.
    No environment logic is executed — always fast.
    """
    return HealthResponse()


@app.post(
    "/reset",
    response_model=Observation,
    tags=["Environment"],
    summary="Start a new episode",
)
def reset(body: ResetRequest) -> Observation:
    """
    Initialise a fresh episode for the given task_id and seed.

    Same seed always produces the identical starting world — fully deterministic.
    Any in-progress episode is discarded and replaced.
    Returns the initial Observation at simulation_hour = 0.

    Valid task_ids: flood_easy · earthquake_medium · compound_hard
    """
    try:
        return env.reset(task_id=body.task_id, seed=body.seed)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reset failed: {exc}")


@app.post(
    "/step",
    response_model=StepResult,
    tags=["Environment"],
    summary="Execute one action — advances simulation by 6 hours",
)
async def step(
    request: Request,
    _schema_hint: dict = Body(
        default=None,
        example={
            "type": "DispatchResource",
            "resource_type": "ambulances",
            "quantity": 2,
            "zone_id": 1,
        },
    ),
) -> StepResult:
    """
    Execute a single action and advance the simulation by 6 simulated hours.

    Send the action directly as the JSON body — the type field is the
    discriminator that selects the correct action class. No wrapper needed.

    Example bodies:
      { "type": "AssessZone",             "zone_id": 2 }
      { "type": "DispatchResource",       "resource_type": "ambulances", "quantity": 2, "zone_id": 3 }
      { "type": "EstablishFieldHospital", "zone_id": 4 }
      { "type": "OpenAlternativeRoute",   "zone_id": 1 }
      { "type": "RequestExternalAid",     "resource_type": "food_trucks" }
      { "type": "PrioritizeZone",         "zone_id": 0, "priority": "CRITICAL" }
      { "type": "CoordinateWithNGO",      "ngo_id": "UNHCR-01", "task": "shelter_setup" }

    Returns StepResult with new Observation, step reward, cumulative reward,
    done flag, and diagnostic info dict.

    Raises 400 if no episode is active or the episode is already done.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=422, detail="Request body must be valid JSON.")

    try:
        action = _action_adapter.validate_python(body)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid action payload: {exc}")

    try:
        obs, reward, cumulative, done, info = env.step(action)
        return StepResult(
            observation       = obs,
            reward            = reward,
            cumulative_reward = cumulative,
            done              = done,
            info              = info,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Step failed: {exc}")


@app.get(
    "/state",
    response_model=EpisodeState,
    tags=["Environment"],
    summary="Get full current episode state",
)
def state() -> EpisodeState:
    """
    Return the complete EpisodeState snapshot for the active episode.

    Read-only — does not advance the simulation or consume a step.
    Safe to call at any point between /reset and episode completion.

    Includes all zone states, current resources, cumulative reward,
    full events log, and both survival rates.

    Raises 400 if no episode has been started yet.
    """
    try:
        return env.get_state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))