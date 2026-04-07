from __future__ import annotations

from typing import List, Tuple

from app.models import (
    DamageLevel,
    NeedType,
    ResourceInventory,
    RoadStatus,
    WeatherCondition,
    ZoneState,
)
from app.simulation import ResourceWave, SimulationEngine
from app.tasks.base_task import BaseTask

# NO-GO zone IDs: Zone 15 (chemical spill) + neighbours within ±2
NO_GO_ZONE_IDS = [13, 14, 15, 16, 17]


class CompoundHardTask(BaseTask):
    """
    Task 3 — compound_hard
    ─────────────────────────────────────────────
    20 zones, 60% roads blocked (12 of 20).
    Earthquake + chemical spill in Zone 15.
    Zones 13–17 are NO-GO (rescue teams permanently lost if sent there).
    Resources arrive in 3 waves (50% / 30% / 20%).
    STORM activates at hour 36 — ambulance capacity −40%.
    External aid only if RequestExternalAid called before hour 12.
    Expected score: 0.15–0.28
    """

    task_id:                str   = "compound_hard"
    baseline_survival_rate: float = 0.48
    max_steps:              int   = 12

    def build(
        self,
        engine: SimulationEngine,
        seed:   int,
    ) -> Tuple[List[ZoneState], ResourceInventory]:

        # ── NO-GO zones ──────────────────────────────────────────────────────
        engine.configure_no_go_zones(NO_GO_ZONE_IDS)

        # ── Resource waves ────────────────────────────────────────────────────
        # Wave 1 (50%) is the initial ResourceInventory returned below.
        # Waves 2 and 3 are deltas delivered by the engine mid-episode.
        engine.configure_resource_waves([
            ResourceWave(
                trigger_hour=24,
                resources={"ambulances": 3, "food_trucks": 4,
                           "field_hospitals": 1, "rescue_teams": 2},
            ),
            ResourceWave(
                trigger_hour=48,
                resources={"ambulances": 2, "food_trucks": 3,
                           "field_hospitals": 1, "rescue_teams": 2},
            ),
        ])

        # ── Weather: STORM from hour 36 ───────────────────────────────────────
        engine.configure_weather_schedule([
            (36, WeatherCondition.STORM),
        ])

        # ── External aid (deadline: must request before hour 12) ─────────────
        engine.configure_external_aid(
            deadline_hour=12,
            payload={"ambulances": 3, "food_trucks": 5,
                     "field_hospitals": 1, "rescue_teams": 2},
        )

        # ── Zones (12 of 20 BLOCKED = 60%) ───────────────────────────────────
        # Blocked zones: 0, 2, 3, 5, 7, 9, 10, 12, 13, 14, 15, 19
        zones = [
            ZoneState(
                zone_id=0, population=4000,
                damage_level=DamageLevel.CRITICAL,
                road_status=RoadStatus.BLOCKED,
                needs=[NeedType.MEDICAL, NeedType.RESCUE],
                casualties=90, served=False, hospital_present=False,
            ),
            ZoneState(
                zone_id=1, population=3200,
                damage_level=DamageLevel.HIGH,
                road_status=RoadStatus.CLEAR,
                needs=[NeedType.MEDICAL, NeedType.FOOD],
                casualties=30, served=False, hospital_present=False,
            ),
            ZoneState(
                zone_id=2, population=2500,
                damage_level=DamageLevel.CRITICAL,
                road_status=RoadStatus.BLOCKED,
                needs=[NeedType.MEDICAL, NeedType.RESCUE, NeedType.SHELTER],
                casualties=60, served=False, hospital_present=False,
            ),
            ZoneState(
                zone_id=3, population=1800,
                damage_level=DamageLevel.MODERATE,
                road_status=RoadStatus.BLOCKED,
                needs=[NeedType.SHELTER, NeedType.FOOD],
                casualties=15, served=False, hospital_present=False,
            ),
            ZoneState(
                zone_id=4, population=3500,
                damage_level=DamageLevel.CRITICAL,
                road_status=RoadStatus.CLEAR,
                needs=[NeedType.MEDICAL, NeedType.RESCUE],
                casualties=80, served=False, hospital_present=False,
            ),
            ZoneState(
                zone_id=5, population=2800,
                damage_level=DamageLevel.HIGH,
                road_status=RoadStatus.BLOCKED,
                needs=[NeedType.FOOD, NeedType.SHELTER],
                casualties=25, served=False, hospital_present=False,
            ),
            ZoneState(
                zone_id=6, population=1500,
                damage_level=DamageLevel.MODERATE,
                road_status=RoadStatus.CLEAR,
                needs=[NeedType.FOOD],
                casualties=5, served=False, hospital_present=False,
            ),
            ZoneState(
                zone_id=7, population=4200,
                damage_level=DamageLevel.CRITICAL,
                road_status=RoadStatus.BLOCKED,
                needs=[NeedType.MEDICAL, NeedType.RESCUE, NeedType.SHELTER],
                casualties=100, served=False, hospital_present=False,
            ),
            ZoneState(
                zone_id=8, population=2000,
                damage_level=DamageLevel.HIGH,
                road_status=RoadStatus.CLEAR,
                needs=[NeedType.MEDICAL, NeedType.FOOD],
                casualties=20, served=False, hospital_present=False,
            ),
            ZoneState(
                zone_id=9, population=3600,
                damage_level=DamageLevel.CRITICAL,
                road_status=RoadStatus.BLOCKED,
                needs=[NeedType.MEDICAL, NeedType.RESCUE],
                casualties=85, served=False, hospital_present=False,
            ),
            ZoneState(
                zone_id=10, population=1200,
                damage_level=DamageLevel.LOW,
                road_status=RoadStatus.BLOCKED,
                needs=[NeedType.FOOD],
                casualties=0, served=False, hospital_present=False,
            ),
            ZoneState(
                zone_id=11, population=2800,
                damage_level=DamageLevel.HIGH,
                road_status=RoadStatus.CLEAR,
                needs=[NeedType.SHELTER, NeedType.FOOD],
                casualties=20, served=False, hospital_present=False,
            ),
            ZoneState(
                zone_id=12, population=3900,
                damage_level=DamageLevel.CRITICAL,
                road_status=RoadStatus.BLOCKED,
                needs=[NeedType.MEDICAL, NeedType.RESCUE, NeedType.SHELTER],
                casualties=95, served=False, hospital_present=False,
            ),
            # ── NO-GO zone (border) ───────────────────────────────────────────
            ZoneState(
                zone_id=13, population=2500,
                damage_level=DamageLevel.CRITICAL,
                road_status=RoadStatus.BLOCKED,
                needs=[NeedType.MEDICAL, NeedType.RESCUE],
                casualties=50, served=False, hospital_present=False,
            ),
            # ── NO-GO zone (border) ───────────────────────────────────────────
            ZoneState(
                zone_id=14, population=1800,
                damage_level=DamageLevel.HIGH,
                road_status=RoadStatus.BLOCKED,
                needs=[NeedType.SHELTER, NeedType.FOOD],
                casualties=20, served=False, hospital_present=False,
            ),
            # ── NO-GO zone (chemical spill epicentre) ────────────────────────
            ZoneState(
                zone_id=15, population=3000,
                damage_level=DamageLevel.CRITICAL,
                road_status=RoadStatus.BLOCKED,
                needs=[NeedType.MEDICAL, NeedType.RESCUE],
                casualties=70, served=False, hospital_present=False,
            ),
            # ── NO-GO zone (border) ───────────────────────────────────────────
            ZoneState(
                zone_id=16, population=2200,
                damage_level=DamageLevel.HIGH,
                road_status=RoadStatus.CLEAR,
                needs=[NeedType.MEDICAL, NeedType.SHELTER],
                casualties=35, served=False, hospital_present=False,
            ),
            # ── NO-GO zone (border) ───────────────────────────────────────────
            ZoneState(
                zone_id=17, population=1500,
                damage_level=DamageLevel.MODERATE,
                road_status=RoadStatus.CLEAR,
                needs=[NeedType.SHELTER, NeedType.FOOD],
                casualties=10, served=False, hospital_present=False,
            ),
            ZoneState(
                zone_id=18, population=3400,
                damage_level=DamageLevel.CRITICAL,
                road_status=RoadStatus.CLEAR,
                needs=[NeedType.MEDICAL, NeedType.RESCUE],
                casualties=75, served=False, hospital_present=False,
            ),
            ZoneState(
                zone_id=19, population=2100,
                damage_level=DamageLevel.HIGH,
                road_status=RoadStatus.BLOCKED,
                needs=[NeedType.FOOD, NeedType.SHELTER],
                casualties=25, served=False, hospital_present=False,
            ),
        ]

        # Wave 1 = 50% of total resources (available at hour 0)
        # Total: ambulances=10, food_trucks=14, field_hospitals=4, rescue_teams=8
        resources = ResourceInventory(
            ambulances=5,
            food_trucks=7,
            field_hospitals=2,
            rescue_teams=4,
        )

        return zones, resources