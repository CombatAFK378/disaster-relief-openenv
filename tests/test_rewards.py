import pytest

from app.models import (
    DamageLevel,
    DispatchResource,
    AssessZone,
    NeedType,
    RoadStatus,
    ZoneState,
)
from app.reward import DUPLICATE_ASSESS_PENALTY, MAX_POSSIBLE_REWARD, RewardCalculator


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def make_zone(
    zone_id:      int   = 0,
    population:   int   = 500,
    damage_level        = DamageLevel.HIGH,
    road_status         = RoadStatus.CLEAR,
    needs               = None,
    casualties:   int   = 0,
    served:       bool  = False,
    hospital:     bool  = False,
) -> ZoneState:
    return ZoneState(
        zone_id          = zone_id,
        population       = population,
        damage_level     = damage_level,
        road_status      = road_status,
        needs            = needs or [NeedType.FOOD],
        casualties       = casualties,
        served           = served,
        hospital_present = hospital,
    )


def baseline_reward(
    calc:                  RewardCalculator,
    pre_zones:             list,
    post_zones:            list,
    action                 = None,
    simulation_hour:       int   = 6,
    done:                  bool  = False,
    overflow_occurred:     bool  = False,
    current_survival_rate: float = 0.70,
    baseline_survival_rate:float = 0.61,
    duplicate_assess:      bool  = False,
) -> float:
    if action is None:
        action = AssessZone(zone_id=0)
    return calc.compute_step_reward(
        pre_zones               = pre_zones,
        post_zones              = post_zones,
        action                  = action,
        simulation_hour         = simulation_hour,
        done                    = done,
        overflow_occurred       = overflow_occurred,
        current_survival_rate   = current_survival_rate,
        baseline_survival_rate  = baseline_survival_rate,
        duplicate_assess        = duplicate_assess,
    )


# ─────────────────────────────────────────────
# Test 1 — people reached gives positive reward
# ─────────────────────────────────────────────

def test_people_reached_reward():
    calc = RewardCalculator(task_id="flood_easy")

    pre  = [make_zone(zone_id=0, population=100, casualties=0, served=False)]
    post = [make_zone(zone_id=0, population=100, casualties=0, served=True)]

    reward = baseline_reward(calc, pre, post)

    assert reward > 0.0, f"Expected positive reward for people reached, got {reward}"


# ─────────────────────────────────────────────
# Test 2 — dispatching to BLOCKED zone fires penalty
# ─────────────────────────────────────────────

def test_blocked_dispatch_penalty():
    calc = RewardCalculator(task_id="flood_easy")

    zone   = make_zone(zone_id=1, road_status=RoadStatus.BLOCKED)
    action = DispatchResource(
        resource_type = "ambulances",
        quantity      = 2,
        zone_id       = 1,
    )

    reward = baseline_reward(calc, [zone], [zone], action=action)

    assert reward < 0.0, f"Expected negative reward for blocked dispatch, got {reward}"


# ─────────────────────────────────────────────
# Test 3 — duplicate assess penalty
# ─────────────────────────────────────────────

def test_duplicate_assess_penalty():
    zone = make_zone(zone_id=0)

    calc_normal    = RewardCalculator(task_id="flood_easy")
    calc_duplicate = RewardCalculator(task_id="flood_easy")

    reward_normal    = baseline_reward(
        calc_normal, [zone], [zone], duplicate_assess=False
    )
    reward_duplicate = baseline_reward(
        calc_duplicate, [zone], [zone], duplicate_assess=True
    )

    diff = reward_normal - reward_duplicate
    expected_min = DUPLICATE_ASSESS_PENALTY / MAX_POSSIBLE_REWARD["flood_easy"] * 0.9
    assert diff >= expected_min, (
        f"Expected duplicate assess to reduce reward by at least {expected_min:.4f}, "
        f"normal={reward_normal:.4f} duplicate={reward_duplicate:.4f} diff={diff:.4f}"
    )


# ─────────────────────────────────────────────
# Test 4 — terminal bonus fires on done=True with no overflow
# ─────────────────────────────────────────────

def test_terminal_bonus_no_overflow():
    calc = RewardCalculator(task_id="flood_easy")

    zone = make_zone(zone_id=0, served=True)

    reward = baseline_reward(
        calc,
        pre_zones              = [zone],
        post_zones             = [zone],
        done                   = True,
        overflow_occurred      = False,
        current_survival_rate  = 0.80,
        baseline_survival_rate = 0.61,
    )

    assert reward > 0.0, (
        f"Expected positive terminal reward (hospital bonus + survival delta), got {reward}"
    )


# ─────────────────────────────────────────────
# Test 5 — overdue penalty fires exactly once
# ─────────────────────────────────────────────

def test_overdue_penalty_fires_once():
    calc = RewardCalculator(task_id="earthquake_medium")

    zone = make_zone(
        zone_id      = 3,
        damage_level = DamageLevel.CRITICAL,
        served       = False,
    )

    penalties = []
    for step_num in range(1, 5):
        r = baseline_reward(
            calc,
            pre_zones       = [zone],
            post_zones      = [zone],
            simulation_hour = step_num * 6,
        )
        penalties.append(r)

    assert 3 in calc._overdue_penalized, (
        "Expected zone_id=3 to appear in _overdue_penalized after 12+ hours unserved"
    )

    negative_steps = [r for r in penalties if r < 0]
    assert len(negative_steps) == 1, (
        f"Expected overdue penalty to fire exactly once, "
        f"but got {len(negative_steps)} negative steps: {penalties}"
    )