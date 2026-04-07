from __future__ import annotations

from typing import List, Tuple

from app.models import (
    DamageLevel,
    NeedType,
    ResourceInventory,
    RoadStatus,
    ZoneState,
)
from app.simulation import SimulationEngine
from app.tasks.base_task import BaseTask


class FloodEasyTask(BaseTask):
    """
    Task 1 — flood_easy
    ─────────────────────────────────────────────
    8 zones, ~10% roads blocked (1 of 8).
    Abundant resources. Needs: mostly FOOD.
    No surprise events. Fully predictable.
    Expected score: 0.65–0.75
    """

    task_id:                str   = "flood_easy"
    baseline_survival_rate: float = 0.61
    max_steps:              int   = 12

    def build(
        self,
        engine: SimulationEngine,
        seed:   int,
    ) -> Tuple[List[ZoneState], ResourceInventory]:
        # No aftershocks, no waves, no weather changes, no NO-GO zones.
        # Engine needs no special configuration for this task.

        zones = [
            ZoneState(
                zone_id=0, population=2500,
                damage_level=DamageLevel.HIGH,
                road_status=RoadStatus.CLEAR,
                needs=[NeedType.FOOD, NeedType.SHELTER],
                casualties=5, served=False, hospital_present=False,
            ),
            ZoneState(
                zone_id=1, population=1200,
                damage_level=DamageLevel.LOW,
                road_status=RoadStatus.CLEAR,
                needs=[NeedType.FOOD],
                casualties=0, served=False, hospital_present=False,
            ),
            ZoneState(
                zone_id=2, population=3000,
                damage_level=DamageLevel.CRITICAL,
                road_status=RoadStatus.CLEAR,
                needs=[NeedType.FOOD, NeedType.MEDICAL],
                casualties=30, served=False, hospital_present=False,
            ),
            ZoneState(
                zone_id=3, population=1800,
                damage_level=DamageLevel.MODERATE,
                road_status=RoadStatus.BLOCKED,   # ← the one blocked road (~10%)
                needs=[NeedType.FOOD],
                casualties=8, served=False, hospital_present=False,
            ),
            ZoneState(
                zone_id=4, population=2200,
                damage_level=DamageLevel.HIGH,
                road_status=RoadStatus.CLEAR,
                needs=[NeedType.FOOD, NeedType.RESCUE],
                casualties=12, served=False, hospital_present=False,
            ),
            ZoneState(
                zone_id=5, population=800,
                damage_level=DamageLevel.LOW,
                road_status=RoadStatus.CLEAR,
                needs=[NeedType.FOOD],
                casualties=0, served=False, hospital_present=False,
            ),
            ZoneState(
                zone_id=6, population=1500,
                damage_level=DamageLevel.MODERATE,
                road_status=RoadStatus.CLEAR,
                needs=[NeedType.FOOD],
                casualties=3, served=False, hospital_present=False,
            ),
            ZoneState(
                zone_id=7, population=2800,
                damage_level=DamageLevel.CRITICAL,
                road_status=RoadStatus.CLEAR,
                needs=[NeedType.FOOD, NeedType.MEDICAL, NeedType.RESCUE],
                casualties=25, served=False, hospital_present=False,
            ),
        ]

        resources = ResourceInventory(
            ambulances=10,
            food_trucks=15,
            field_hospitals=4,
            rescue_teams=8,
        )

        return zones, resources