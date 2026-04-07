import pytest

from app.environment import DisasterEnv
from app.models import (
    AssessZone,
    DispatchResource,
    Observation,
)


@pytest.fixture
def env():
    """Fresh DisasterEnv instance for each test."""
    return DisasterEnv()


# ─────────────────────────────────────────────
# Test 1 — reset returns valid Observation
# ─────────────────────────────────────────────

def test_reset_flood_easy(env):
    obs = env.reset("flood_easy", 42)

    assert isinstance(obs, Observation)
    assert len(obs.zones) == 8
    assert obs.simulation_hour == 0
    assert obs.baseline_survival_rate == 0.61
    assert 0.0 <= obs.current_survival_rate <= 1.0
    assert obs.resources.ambulances >= 0
    assert obs.resources.food_trucks >= 0


# ─────────────────────────────────────────────
# Test 2 — determinism: same seed = same world
# ─────────────────────────────────────────────

def test_reset_determinism(env):
    obs_a = env.reset("flood_easy", 42)
    obs_b = env.reset("flood_easy", 42)

    zone_ids_a     = [z.zone_id      for z in obs_a.zones]
    zone_ids_b     = [z.zone_id      for z in obs_b.zones]
    damage_levels_a = [z.damage_level for z in obs_a.zones]
    damage_levels_b = [z.damage_level for z in obs_b.zones]

    assert zone_ids_a      == zone_ids_b
    assert damage_levels_a == damage_levels_b

    # Different seed must produce a different world
    obs_c = env.reset("earthquake_medium", 99)
    assert len(obs_c.zones) == 15


# ─────────────────────────────────────────────
# Test 3 — step returns reward in valid range
# ─────────────────────────────────────────────

def test_step_returns_valid_reward(env):
    env.reset("flood_easy", 42)

    action = AssessZone(zone_id=0)
    obs, reward, cumulative, done, info = env.step(action)

    assert isinstance(obs, Observation)
    assert -1.0 <= reward     <= 1.0
    assert  0.0 <= cumulative <= 1.0
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert obs.simulation_hour == 6


# ─────────────────────────────────────────────
# Test 4 — done == True after max_steps (12)
# ─────────────────────────────────────────────

def test_done_after_max_steps(env):
    env.reset("flood_easy", 42)

    done = False
    for i in range(12):
        # Use zone_id=0 for all steps — duplicate penalty fires but that is fine
        _, _, _, done, _ = env.step(AssessZone(zone_id=0))

    assert done is True


# ─────────────────────────────────────────────
# Test 5 — GET /state matches observation after step
# ─────────────────────────────────────────────

def test_state_matches_after_step(env):
    env.reset("flood_easy", 42)

    action = DispatchResource(resource_type="ambulances", quantity=2, zone_id=0)
    obs, _, _, _, _ = env.step(action)

    state = env.get_state()

    assert state.simulation_hour       == obs.simulation_hour
    assert state.current_survival_rate == obs.current_survival_rate
    assert state.task_id               == "flood_easy"
    assert len(state.zones)            == len(obs.zones)
    assert isinstance(state.events_log, list)
    assert len(state.events_log)       >= 1


# ─────────────────────────────────────────────
# Test 6 — all 3 tasks load without error
# ─────────────────────────────────────────────

def test_all_tasks_load(env):
    tasks = [
        ("flood_easy",        8),
        ("earthquake_medium", 15),
        ("compound_hard",     20),
    ]
    for task_id, expected_zones in tasks:
        obs = env.reset(task_id, 42)
        assert len(obs.zones) == expected_zones, (
            f"{task_id}: expected {expected_zones} zones, got {len(obs.zones)}"
        )
        assert obs.simulation_hour == 0
        assert obs.baseline_survival_rate > 0.0