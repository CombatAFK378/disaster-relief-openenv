from __future__ import annotations

from typing import List, Tuple

from app.models import (
    DamageLevel,
    NeedType,
    ResourceInventory,
    RoadStatus,
    ZoneState,
)
from app.simulation import AfterShockEvent, SimulationEngine
from app.tasks.base_task import BaseTask


class EarthquakeMediumTask(BaseTask):
    """
    Task 2 — earthquake_medium
    ─────────────────────────────────────────────
    15 zones, 40% roads blocked (6 of 15).
    Mixed needs: MEDICAL + SHELTER + FOOD.
    Aftershocks at hour 18 and hour 36, each blocks 2 more roads.
    Expected score: 0.40–0.55
    """

    task_id:                str   = "earthquake_medium"
    baseline_survival_rate: float = 0.54
    max_steps:              int   = 12

    def build(
        self,
        engine: SimulationEngine,
        seed:   int,
    ) -> Tuple[List[ZoneState], ResourceInventory]:
        # Two aftershocks — road blockings are seeded via engine.rng
        engine.configure_aftershocks([
            AfterShockEvent(trigger_hour=18, roads_to_block=2),
            AfterShockEvent(trigger_hour=36, roads_to_block=2),
        ])

        # 6 of 15 zones have BLOCKED roads = 40%
        zones = [
            ZoneState(
                zone_id=0, population=3500,
                damage_level=DamageLevel.CRITICAL,
                road_status=RoadStatus.CLEAR,
                needs=[NeedType.MEDICAL, NeedType.RESCUE],
                casualties=45, served=False, hospital_present=False,
            ),
            ZoneState(
                zone_id=1, population=2800,
                damage_level=DamageLevel.HIGH,
                road_status=RoadStatus.CLEAR,
                needs=[NeedType.MEDICAL, NeedType.FOOD],
                casualties=20, served=False, hospital_present=False,
            ),
            ZoneState(
                zone_id=2, population=1500,
                damage_level=DamageLevel.MODERATE,
                road_status=RoadStatus.BLOCKED,
                needs=[NeedType.SHELTER],
                casualties=5, served=False, hospital_present=False,
            ),
            ZoneState(
                zone_id=3, population=4000,
                damage_level=DamageLevel.CRITICAL,
                road_status=RoadStatus.BLOCKED,
                needs=[NeedType.MEDICAL, NeedType.RESCUE, NeedType.SHELTER],
                casualties=80, served=False, hospital_present=False,
            ),
            ZoneState(
                zone_id=4, population=2200,
                damage_level=DamageLevel.HIGH,
                road_status=RoadStatus.CLEAR,
                needs=[NeedType.FOOD, NeedType.SHELTER],
                casualties=15, served=False, hospital_present=False,
            ),
            ZoneState(
                zone_id=5, population=1800,
                damage_level=DamageLevel.LOW,
                road_status=RoadStatus.CLEAR,
                needs=[NeedType.FOOD],
                casualties=0, served=False, hospital_present=False,
            ),
            ZoneState(
                zone_id=6, population=3200,
                damage_level=DamageLevel.CRITICAL,
                road_status=RoadStatus.CLEAR,
                needs=[NeedType.MEDICAL, NeedType.RESCUE],
                casualties=60, served=False, hospital_present=False,
            ),
            ZoneState(
                zone_id=7, population=2500,
                damage_level=DamageLevel.HIGH,
                road_status=RoadStatus.BLOCKED,
                needs=[NeedType.SHELTER, NeedType.FOOD],
                casualties=30, served=False, hospital_present=False,
            ),
            ZoneState(
                zone_id=8, population=1200,
                damage_level=DamageLevel.MODERATE,
                road_status=RoadStatus.CLEAR,
                needs=[NeedType.FOOD],
                casualties=5, served=False, hospital_present=False,
            ),
            ZoneState(
                zone_id=9, population=3800,
                damage_level=DamageLevel.CRITICAL,
                road_status=RoadStatus.BLOCKED,
                needs=[NeedType.MEDICAL, NeedType.RESCUE, NeedType.SHELTER],
                casualties=70, served=False, hospital_present=False,
            ),
            ZoneState(
                zone_id=10, population=2000,
                damage_level=DamageLevel.HIGH,
                road_status=RoadStatus.CLEAR,
                needs=[NeedType.FOOD, NeedType.SHELTER],
                casualties=10, served=False, hospital_present=False,
            ),
            ZoneState(
                zone_id=11, population=1600,
                damage_level=DamageLevel.MODERATE,
                road_status=RoadStatus.BLOCKED,
                needs=[NeedType.SHELTER],
                casualties=8, served=False, hospital_present=False,
            ),
            ZoneState(
                zone_id=12, population=2900,
                damage_level=DamageLevel.HIGH,
                road_status=RoadStatus.CLEAR,
                needs=[NeedType.MEDICAL, NeedType.FOOD],
                casualties=25, served=False, hospital_present=False,
            ),
            ZoneState(
                zone_id=13, population=3100,
                damage_level=DamageLevel.CRITICAL,
                road_status=RoadStatus.BLOCKED,
                needs=[NeedType.MEDICAL, NeedType.RESCUE],
                casualties=55, served=False, hospital_present=False,
            ),
            ZoneState(
                zone_id=14, population=1900,
                damage_level=DamageLevel.MODERATE,
                road_status=RoadStatus.CLEAR,
                needs=[NeedType.FOOD, NeedType.SHELTER],
                casualties=10, served=False, hospital_present=False,
            ),
        ]

        resources = ResourceInventory(
            ambulances=6,
            food_trucks=8,
            field_hospitals=3,
            rescue_teams=5,
        )

        return zones, resources