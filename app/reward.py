from __future__ import annotations

from typing import Dict, List, Set

from app.models import (
    Action,
    DamageLevel,
    DispatchResource,
    RoadStatus,
    ZoneState,
)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

HOURS_PER_STEP           = 6
PEOPLE_REWARD_RATE       = 0.02
PEOPLE_CAP_PER_STEP      = 500
CRITICAL_CLEAR_BONUS     = 5.0
HOSPITAL_BONUS           = 10.0
TERMINAL_SURVIVAL_WEIGHT = 15.0
BLOCKED_DISPATCH_PENALTY = 2.0
CRITICAL_OVERDUE_PENALTY = 2.0
DUPLICATE_ASSESS_PENALTY = 1.0
CRITICAL_OVERDUE_HOURS   = 12   # hours unserved before penalty fires

# Normalization denominators — theoretical max raw reward per task
MAX_POSSIBLE_REWARD: Dict[str, float] = {
    "flood_easy":        58.0,
    "earthquake_medium": 127.0,
    "compound_hard":    274.0,
}


# ─────────────────────────────────────────────
# RewardCalculator
# ─────────────────────────────────────────────

class RewardCalculator:
    """
    Stateful reward calculator — holds per-episode tracking data.
    Instantiated once per episode by DisasterEnv.reset().

    Responsibilities:
      - Track cumulative unserved hours per CRITICAL zone
      - Track which zones have already triggered the overdue penalty
      - Compute and normalize the step reward

    Never calls simulation.py, never calls an LLM.
    Receives all needed state as parameters.
    """

    def __init__(self, task_id: str) -> None:
        self.task_id     = task_id
        self._max_reward = MAX_POSSIBLE_REWARD.get(task_id, 10.0)

        # zone_id → cumulative hours spent as CRITICAL + unserved
        self._unserved_critical_hours: Dict[int, int] = {}

        # zone_ids that have already triggered the overdue penalty (fire once only)
        self._overdue_penalized: Set[int] = set()

    def reset(self) -> None:
        """Call at the start of every new episode."""
        self._unserved_critical_hours = {}
        self._overdue_penalized       = set()

    # ── Main entry point ──────────────────────────────────────────────────────

    def compute_step_reward(
        self,
        pre_zones:              List[ZoneState],
        post_zones:             List[ZoneState],
        action:                 Action,
        simulation_hour:        int,    # hour value AFTER this step (e.g. 6, 12, …)
        done:                   bool,
        overflow_occurred:      bool,
        current_survival_rate:  float,
        baseline_survival_rate: float,
        duplicate_assess:       bool = False,
    ) -> float:
        """
        Compute the normalised reward for one step.

        Parameters
        ----------
        pre_zones              Zone states BEFORE the action was applied.
        post_zones             Zone states AFTER the action was applied.
        action                 The typed action taken this step.
        simulation_hour        Simulation clock after this step (multiples of 6).
        done                   True if this is the terminal step.
        overflow_occurred      True if any hospital exceeded capacity this episode.
        current_survival_rate  Survival rate computed from post_zones.
        baseline_survival_rate Hardcoded constant for this task.
        duplicate_assess       True if this AssessZone targeted an already-assessed zone.

        Returns
        -------
        float  Normalised step reward in roughly [−1.0, 1.0].
               Cumulative clamped to [0.0, 1.0] at episode end by DisasterEnv.
        """
        raw     = 0.0
        pre_map = {z.zone_id: z for z in pre_zones}

        # ── Step 1: update unserved-critical-hours tracker ────────────────────
        self._update_unserved_hours(post_zones)

        # ── Component 1: people reached this step  (+0.02 × reached, cap 500) ─
        raw += self._people_reached_reward(pre_map, post_zones)

        # ── Component 2: CRITICAL zone cleared within 24 hours  (+0.10 each) ──
        raw += self._critical_cleared_reward(pre_map, post_zones, simulation_hour)

        # ── Component 5: resource dispatched to BLOCKED zone  (−0.10) ─────────
        raw -= self._blocked_dispatch_penalty(action, pre_map)

        # ── Component 6: CRITICAL zone overdue > 12 h  (−0.20, fires once) ───
        raw -= self._overdue_penalty()

        # ── Component 7: duplicate AssessZone  (−0.05) ───────────────────────
        if duplicate_assess:
            raw -= DUPLICATE_ASSESS_PENALTY

        # ── Components 3 & 4: terminal bonuses (only when done=True) ─────────
        if done:
            raw += self._terminal_bonuses(
                overflow_occurred,
                current_survival_rate,
                baseline_survival_rate,
            )

        # ── Normalise by theoretical max for this task ────────────────────────
        return raw / self._max_reward

    # ── Private helpers ───────────────────────────────────────────────────────

    def _update_unserved_hours(self, post_zones: List[ZoneState]) -> None:
        """
        Accumulate HOURS_PER_STEP for every CRITICAL+unserved zone.
        Reset counter to 0 when a zone becomes served.
        """
        for z in post_zones:
            if z.damage_level == DamageLevel.CRITICAL and not z.served:
                prev = self._unserved_critical_hours.get(z.zone_id, 0)
                self._unserved_critical_hours[z.zone_id] = prev + HOURS_PER_STEP
            else:
                # Served or no longer CRITICAL — clear the counter
                self._unserved_critical_hours[z.zone_id] = 0

    def _people_reached_reward(
        self,
        pre_map:   Dict[int, ZoneState],
        post_zones: List[ZoneState],
    ) -> float:
        """
        +0.02 per person in zones that transitioned from unserved → served,
        capped at 500 people total per step.
        People = population − existing casualties (living survivors reached).
        """
        people_reached = 0
        for z_post in post_zones:
            z_pre = pre_map.get(z_post.zone_id)
            if z_pre and not z_pre.served and z_post.served:
                living = max(0, z_post.population - z_post.casualties)
                people_reached += living

        people_reached = min(people_reached, PEOPLE_CAP_PER_STEP)
        return PEOPLE_REWARD_RATE * people_reached

    def _critical_cleared_reward(
        self,
        pre_map:         Dict[int, ZoneState],
        post_zones:      List[ZoneState],
        simulation_hour: int,
    ) -> float:
        """
        +0.10 for each CRITICAL zone that becomes served within the first 24 hours.
        A zone qualifies if it was CRITICAL+unserved before and served after,
        and simulation_hour (post-step clock) is <= 24.
        """
        reward = 0.0
        for z_post in post_zones:
            z_pre = pre_map.get(z_post.zone_id)
            if (
                z_pre
                and z_pre.damage_level == DamageLevel.CRITICAL
                and not z_pre.served
                and z_post.served
                and simulation_hour <= 24
            ):
                reward += CRITICAL_CLEAR_BONUS
        return reward

    def _blocked_dispatch_penalty(
        self,
        action:  Action,
        pre_map: Dict[int, ZoneState],
    ) -> float:
        """
        −0.10 when a DispatchResource action targets a zone whose road was
        BLOCKED before the action. Returns penalty magnitude (positive float).
        """
        if not isinstance(action, DispatchResource):
            return 0.0
        target = pre_map.get(action.zone_id)
        if target and target.road_status == RoadStatus.BLOCKED:
            return BLOCKED_DISPATCH_PENALTY
        return 0.0

    def _overdue_penalty(self) -> float:
        """
        −0.20 for each CRITICAL zone that has now been unserved for > 12 hours.
        Fires exactly once per zone (tracked in _overdue_penalized).
        Returns penalty magnitude (positive float).
        """
        penalty = 0.0
        for zone_id, hours in self._unserved_critical_hours.items():
            if hours > CRITICAL_OVERDUE_HOURS and zone_id not in self._overdue_penalized:
                penalty += CRITICAL_OVERDUE_PENALTY
                self._overdue_penalized.add(zone_id)
        return penalty

    def _terminal_bonuses(
        self,
        overflow_occurred:      bool,
        current_survival_rate:  float,
        baseline_survival_rate: float,
    ) -> float:
        """
        Component 3: +0.15 if hospital capacity was never exceeded.
        Component 4: +0.20 × (agent_survival_rate − baseline_survival_rate).
                     Clamped to 0 if agent performed worse than baseline.
        """
        bonus = 0.0

        if not overflow_occurred:
            bonus += HOSPITAL_BONUS

        survival_delta = max(0.0, current_survival_rate - baseline_survival_rate)
        bonus += TERMINAL_SURVIVAL_WEIGHT * survival_delta

        return bonus