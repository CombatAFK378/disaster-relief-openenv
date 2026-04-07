---
title: Disaster Relief Coordinator
emoji: 🚨
colorFrom: red
colorTo: orange
sdk: docker
app_port: 8000
tags:
  - openenv
  - disaster-relief
  - humanitarian
  - reinforcement-learning
---

# 🚨 Disaster Relief Coordinator — OpenEnv

![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-compliant-success)
![HuggingFace Spaces](https://img.shields.io/badge/HuggingFace-Spaces-orange)
![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)
![MIT License](https://img.shields.io/badge/License-MIT-lightgrey)

**An OpenEnv-compliant disaster response environment where an AI coordinator allocates emergency resources across flood, earthquake, and compound-crisis scenarios under strict time constraints.**

Repository: https://github.com/CombatAFK/disaster-relief-openenv

Hugging Face Space: https://huggingface.co/spaces/CombatAFK/disaster-relief-openenv

## Why This Exists

Disaster response teams must make high-stakes allocation decisions with incomplete information, constrained logistics, and rapidly changing field conditions. This environment models those pressures with deterministic simulation and explicit reward accounting so policies can be evaluated consistently.

The scenario framing is aligned with humanitarian coordination realities (multi-zone triage, route failures, aftershocks, external aid timing), reflecting operational concerns associated with UN OCHA-style incident management where delays in routing and misallocation directly affect survival outcomes.

## Tasks Overview

| Task ID | Difficulty | Zones | Roads Blocked | Special Events | Expected Score |
|---|---:|---:|---:|---|---|
| flood_easy | Easy | 8 | 1/8 (10%) | No aftershocks, no weather shifts, no resource waves | 0.65–0.75 |
| earthquake_medium | Medium | 15 | 6/15 (40%) | Aftershocks at hour 18 and 36 (each blocks 2 additional roads) | 0.40–0.55 |
| compound_hard | Hard | 20 | 12/20 (60%) | NO-GO zones (13–17), resource waves (hour 24, 48), STORM at hour 36, external aid deadline at hour 12 | 0.15–0.28 |

## Observation Space

Observation object is defined by `Observation` in `app/models.py`.

| Field | Type | Description |
|---|---|---|
| zones | List[ZoneState] | Per-zone state list |
| resources | ResourceInventory | Current available resource counts |
| simulation_hour | int | Simulation clock in hours |
| weather | WeatherState | Current weather and affected zone IDs |
| active_events | List[str] | Recent event log entries |
| baseline_survival_rate | float | Task baseline survival reference |
| current_survival_rate | float | Live survival rate for current state |

ZoneState structure:

| Field | Type | Description |
|---|---|---|
| zone_id | int | Unique zone identifier |
| population | int | Zone population |
| damage_level | DamageLevel | CRITICAL / HIGH / MODERATE / LOW |
| road_status | RoadStatus | CLEAR / DEGRADED / BLOCKED |
| needs | List[NeedType] | MEDICAL / FOOD / SHELTER / RESCUE |
| casualties | int | Current casualties in the zone |
| served | bool | Whether zone has been served |
| hospital_present | bool | Whether a field hospital is present |

## Action Space

The API action payload is a discriminated union (`Action`) across 7 action classes in `app/models.py`.

| Action | Parameters | Description |
|---|---|---|
| AssessZone | zone_id:int | Gather current details for a zone |
| DispatchResource | resource_type, quantity:int, zone_id:int | Send resources to a target zone |
| EstablishFieldHospital | zone_id:int | Place one field hospital in a zone |
| OpenAlternativeRoute | zone_id:int | Improve road access for one zone |
| RequestExternalAid | resource_type | Request aid package (deadline-sensitive in compound_hard) |
| PrioritizeZone | zone_id:int, priority | Set planning priority marker |
| CoordinateWithNGO | ngo_id:str, task:str | Register NGO coordination task |

JSON examples (one per action):

```json
{"type": "AssessZone", "zone_id": 2}
```

```json
{"type": "DispatchResource", "resource_type": "ambulances", "quantity": 2, "zone_id": 3}
```

```json
{"type": "EstablishFieldHospital", "zone_id": 4}
```

```json
{"type": "OpenAlternativeRoute", "zone_id": 1}
```

```json
{"type": "RequestExternalAid", "resource_type": "food_trucks"}
```

```json
{"type": "PrioritizeZone", "zone_id": 0, "priority": "CRITICAL"}
```

```json
{"type": "CoordinateWithNGO", "ngo_id": "UNHCR-01", "task": "shelter_setup"}
```

## Reward Function

Reward logic is implemented in `app/reward.py` via `RewardCalculator.compute_step_reward`.

| Component | Value | Fires When |
|---|---|---|
| People reached bonus | +0.02 per person (cap 500 people/step) | A zone transitions unserved → served |
| CRITICAL cleared bonus | +0.10 per zone | A CRITICAL unserved zone is served by hour <= 24 |
| No-overflow hospital bonus | +0.15 | Terminal step, if hospital overflow never occurred |
| Survival improvement bonus | +0.20 × max(0, survival_delta) | Terminal step |
| Blocked dispatch penalty | -0.10 | DispatchResource targets BLOCKED road |
| Critical overdue penalty | -0.20 (once/zone) | CRITICAL zone remains unserved for > 12 hours |
| Duplicate assess penalty | -0.05 | AssessZone repeated on already-assessed zone |

Per-step raw reward is normalized by task-specific maxima (`MAX_POSSIBLE_REWARD`) and cumulative reward is clamped to [0.00, 1.00] at episode end in `app/environment.py`.

## Setup & Installation

```bash
git clone https://github.com/CombatAFK/disaster-relief-openenv
cd disaster-relief-openenv
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Docker Usage

```bash
docker build -t disaster-env .
docker run -p 8000:8000 disaster-env
```

## Running Tests

```bash
pytest tests/ -v
```

Expected:

- 21 tests total, all passing
- tests/test_environment.py: 6 tests
- tests/test_rewards.py: 5 tests
- tests/test_tasks.py: 10 tests

## Running Inference Baseline

Use two terminals.

Terminal 1 (environment server):

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Terminal 2 (agent runner):

```bash
python inference.py
```

Sample log format:

```text
[START] task=flood_easy env=disaster-relief-openenv model=openai/gpt-oss-120b
[STEP] step=1 action=DispatchResource reward=0.17 done=false error=null
[END] success=true steps=12 score=0.70 rewards=0.17,0.05,0.00,0.07,0.06,0.08,0.04,0.03,0.05,0.06,0.04,0.05
```

## Baseline Scores

| Task ID | Agent Baseline Score | No-AI Baseline (task baseline_survival_rate) |
|---|---:|---:|
| flood_easy | 0.70 | 0.61 |
| earthquake_medium | 0.47 | 0.54 |
| compound_hard | 0.21 | 0.48 |

## Project Structure

```text
disaster-relief-openenv/
├── .env.example
├── .gitignore
├── conftest.py
├── Dockerfile
├── inference.py
├── openenv.yaml
├── pyproject.toml
├── pytest.ini
├── README.md
├── requirements.txt
├── uv.lock
├── app/
│   ├── __init__.py
│   ├── environment.py
│   ├── main.py
│   ├── models.py
│   ├── reward.py
│   ├── simulation.py
│   └── tasks/
│       ├── __init__.py
│       ├── base_task.py
│       ├── compound_hard.py
│       ├── earthquake_medium.py
│       └── flood_easy.py
├── server/
│   ├── __init__.py
│   └── app.py
└── tests/
		├── __init__.py
		├── test_environment.py
		├── test_rewards.py
		└── test_tasks.py
```

## HF Spaces Deployment

This repository is configured for Docker-based Space deployment (`sdk: docker`, `app_port: 8000`). Keep the YAML front matter block at the top of this README when publishing to Hugging Face Spaces.

Space URL placeholder:

https://huggingface.co/spaces/CombatAFK/disaster-relief-openenv

## License

MIT
