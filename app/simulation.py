from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from app.models import (
    DamageLevel,
    ResourceInventory,
    RoadStatus,
    WeatherCondition,
    WeatherState,
    ZoneState,
)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

HOURS_PER_STEP          = 6
MAX_STEPS               = 12
HOSPITAL_CAPACITY       = 200
CRITICAL_CASUALTY_RATE  = 0.02
STORM_AMBULANCE_PENALTY = 0.40


# ─────────────────────────────────────────────
# Internal tracking dataclasses
# ─────────────────────────────────────────────

@dataclass
class DispatchRecord:
    resource_type:    str
    quantity:         int
    zone_id:          int
    return_step:      int
    permanently_lost: bool = False


@dataclass
class AfterShockEvent:
    trigger_hour:   int
    roads_to_block: int


@dataclass
class ResourceWave:
    trigger_hour: int
    resources:    Dict[str, int]


# ─────────────────────────────────────────────
# Pure helper functions
# ─────────────────────────────────────────────

def compute_step_casualties(zone: ZoneState) -> int:
    """
    Casualties added in one 6-hour step for a single unserved CRITICAL zone.
    Formula: at_risk_population × 0.02 × 6
    Returns 0 for served zones or non-CRITICAL damage levels.
    """
    if zone.served or zone.damage_level != DamageLevel.CRITICAL:
        return 0
    at_risk = max(0, zone.population - zone.casualties)
    return int(at_risk * CRITICAL_CASUALTY_RATE * HOURS_PER_STEP)


def compute_survival_rate(zones: List[ZoneState]) -> float:
    """
    (total_population - total_casualties) / total_population.
    Clamped to [0.0, 1.0]. Returns 1.0 when total population is zero.
    """
    total_pop = sum(z.population for z in zones)
    if total_pop == 0:
        return 1.0
    total_casualties = sum(z.casualties for z in zones)
    return max(0.0, min(1.0, (total_pop - total_casualties) / total_pop))


def effective_ambulance_count(base_count: int, weather: WeatherState) -> int:
    """Returns operationally available ambulances after weather penalties."""
    if weather.condition == WeatherCondition.STORM:
        return max(0, int(base_count * (1.0 - STORM_AMBULANCE_PENALTY)))
    return base_count


# ─────────────────────────────────────────────
# SimulationEngine
# ─────────────────────────────────────────────

class SimulationEngine:
    """
    Owns all simulation physics. No network calls. No FastAPI imports.
    All randomness routed through self.rng (seeded at construction).
    Called exclusively by DisasterEnv (environment.py).
    """

    def __init__(self, seed: int) -> None:
        self.rng: random.Random = random.Random(seed)

        # Dispatch tracking
        self._dispatched: List[DispatchRecord] = []

        # Hospital state
        self._hospital_patients:          Dict[int, int] = {}
        self._hospital_overflow_occurred: bool           = False

        # Task-specific configuration
        self._no_go_zones:      Set[int]                            = set()
        self._aftershocks:      List[AfterShockEvent]               = []
        self._resource_waves:   List[ResourceWave]                  = []
        self._weather_schedule: List[Tuple[int, WeatherCondition]]  = []

        # External aid — FIX 1: added _aid_arrival_step tracker
        self._aid_deadline_hour:  Optional[int]      = None
        self._aid_payload:        Dict[str, int]     = {}
        self._aid_requested:      bool               = False
        self._aid_delivered:      bool               = False
        self._aid_arrival_step:   Optional[int]      = None   # FIX 1

        # Assessed zones
        self._assessed_zones: Set[int] = set()

    # ── Configuration ─────────────────────────────────────────────────────────

    def configure_no_go_zones(self, zone_ids: List[int]) -> None:
        self._no_go_zones.update(zone_ids)

    def configure_aftershocks(self, events: List[AfterShockEvent]) -> None:
        self._aftershocks = sorted(events, key=lambda e: e.trigger_hour)

    def configure_resource_waves(self, waves: List[ResourceWave]) -> None:
        self._resource_waves = sorted(waves, key=lambda w: w.trigger_hour)

    def configure_weather_schedule(
        self, schedule: List[Tuple[int, WeatherCondition]]
    ) -> None:
        self._weather_schedule = sorted(schedule, key=lambda t: t[0])

    def configure_external_aid(
        self,
        deadline_hour: Optional[int],
        payload: Dict[str, int],
    ) -> None:
        self._aid_deadline_hour = deadline_hour
        self._aid_payload       = payload

    # ── Queries ───────────────────────────────────────────────────────────────

    def is_no_go(self, zone_id: int) -> bool:
        return zone_id in self._no_go_zones

    def is_assessed(self, zone_id: int) -> bool:
        return zone_id in self._assessed_zones

    def mark_assessed(self, zone_id: int) -> None:
        self._assessed_zones.add(zone_id)

    def hospital_overflow_occurred(self) -> bool:
        return self._hospital_overflow_occurred

    def get_hospital_patients(self, zone_id: int) -> int:
        return self._hospital_patients.get(zone_id, 0)

    # ── Resource Availability ─────────────────────────────────────────────────

    def available_quantity(
        self,
        resource_type:    str,
        total_inventory:  ResourceInventory,   # FIX 2: renamed from base_inventory
    ) -> int:
        """
        Units currently at base = total − dispatched (excluding permanently lost).
        resource_type must match a ResourceInventory field name.
        """
        total  = getattr(total_inventory, resource_type, 0)   # FIX 2: renamed
        in_use = sum(
            r.quantity for r in self._dispatched
            if r.resource_type == resource_type and not r.permanently_lost
        )
        return max(0, total - in_use)

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def dispatch(
        self,
        resource_type:   str,
        quantity:        int,
        zone_id:         int,
        current_step:    int,
        total_inventory: ResourceInventory,   # FIX 2: renamed from base_inventory
    ) -> Tuple[bool, str]:
        """
        Record a resource deployment. Returns (success, message).
        rescue_teams → NO-GO zone: teams are permanently lost.
        """
        if resource_type == "rescue_teams" and self.is_no_go(zone_id):
            self._dispatched.append(
                DispatchRecord(
                    resource_type=resource_type,
                    quantity=quantity,
                    zone_id=zone_id,
                    return_step=-1,
                    permanently_lost=True,
                )
            )
            return True, f"WARNING: rescue_teams dispatched to NO-GO zone {zone_id} — permanently lost"

        avail = self.available_quantity(resource_type, total_inventory)   # FIX 2
        if avail < quantity:
            return False, f"insufficient {resource_type}: requested {quantity}, available {avail}"

        self._dispatched.append(
            DispatchRecord(
                resource_type=resource_type,
                quantity=quantity,
                zone_id=zone_id,
                return_step=current_step + 2,
            )
        )
        return True, "dispatched"

    def collect_returned_resources(self, current_step: int) -> Dict[str, int]:
        """
        Free dispatch records whose return_step <= current_step.
        Returns dict of resource_type → quantity freed.
        """
        freed:     Dict[str, int]       = {}
        remaining: List[DispatchRecord] = []

        for rec in self._dispatched:
            if not rec.permanently_lost and current_step >= rec.return_step:
                freed[rec.resource_type] = freed.get(rec.resource_type, 0) + rec.quantity
            else:
                remaining.append(rec)

        self._dispatched = remaining
        return freed

    # ── Hospital ──────────────────────────────────────────────────────────────

    def init_hospital(self, zone_id: int) -> None:
        self._hospital_patients.setdefault(zone_id, 0)

    def admit_patients(self, zone_id: int, count: int) -> Tuple[int, bool]:
        """
        Admit patients up to HOSPITAL_CAPACITY.
        Returns (actually_admitted, overflow_occurred).
        """
        current  = self._hospital_patients.get(zone_id, 0)
        space    = max(0, HOSPITAL_CAPACITY - current)
        admitted = min(count, space)
        overflow = count > space

        self._hospital_patients[zone_id] = current + admitted
        if overflow:
            self._hospital_overflow_occurred = True

        return admitted, overflow

    # ── Casualties ────────────────────────────────────────────────────────────

    def apply_step_casualties(
        self, zones: List[ZoneState]
    ) -> Tuple[List[ZoneState], List[str]]:
        """
        Accumulate casualties for every unserved CRITICAL zone.
        Returns (updated_zones, event_messages).
        """
        updated:  List[ZoneState] = []
        messages: List[str]       = []

        for z in zones:
            delta = compute_step_casualties(z)
            if delta > 0:
                new_casualties = z.casualties + delta
                updated.append(z.model_copy(update={"casualties": new_casualties}))
                messages.append(
                    f"CASUALTIES: zone {z.zone_id} +{delta} (total {new_casualties})"
                )
            else:
                updated.append(z)

        return updated, messages

    # ── Weather ───────────────────────────────────────────────────────────────

    def get_weather_at_hour(self, simulation_hour: int) -> WeatherCondition:
        """Last scheduled change at or before current hour wins."""
        condition = WeatherCondition.CLEAR
        for (hour, new_condition) in self._weather_schedule:
            if simulation_hour >= hour:
                condition = new_condition
        return condition

    def build_weather_state(
        self,
        simulation_hour: int,
        zones: List[ZoneState],
    ) -> WeatherState:
        condition = self.get_weather_at_hour(simulation_hour)

        if condition == WeatherCondition.STORM:
            affects = [z.zone_id for z in zones]
        elif condition == WeatherCondition.RAIN:
            affects = [
                z.zone_id for z in zones
                if z.damage_level in (DamageLevel.CRITICAL, DamageLevel.HIGH)
            ]
        else:
            affects = []

        return WeatherState(condition=condition, affects_zones=affects)

    # ── Aftershocks ───────────────────────────────────────────────────────────

    def get_aftershock_at_hour(self, simulation_hour: int) -> Optional[AfterShockEvent]:
        for event in self._aftershocks:
            if event.trigger_hour == simulation_hour:
                return event
        return None

    def apply_aftershock(
        self,
        event: AfterShockEvent,
        zones: List[ZoneState],
    ) -> Tuple[List[ZoneState], List[str]]:
        """
        Randomly block `event.roads_to_block` non-already-BLOCKED roads.
        Uses self.rng for full determinism.
        """
        candidates  = [z for z in zones if z.road_status != RoadStatus.BLOCKED]
        sample_size = min(event.roads_to_block, len(candidates))
        to_block    = {z.zone_id for z in self.rng.sample(candidates, sample_size)}

        updated:  List[ZoneState] = []
        messages: List[str]       = []

        for z in zones:
            if z.zone_id in to_block:
                updated.append(z.model_copy(update={"road_status": RoadStatus.BLOCKED}))
                messages.append(f"AFTERSHOCK: road in zone {z.zone_id} blocked")
            else:
                updated.append(z)

        return updated, messages

    # ── Resource Waves ────────────────────────────────────────────────────────

    def get_wave_at_hour(self, simulation_hour: int) -> Optional[ResourceWave]:
        for wave in self._resource_waves:
            if wave.trigger_hour == simulation_hour:
                return wave
        return None

    # ── External Aid ──────────────────────────────────────────────────────────

    def request_external_aid(
        self,
        simulation_hour: int,
        current_step:    int,    # FIX 1: added parameter
    ) -> Tuple[bool, str]:
        """
        Register an external aid request.
        FIX 1: sets _aid_arrival_step = current_step + 1 so aid
        cannot arrive in the same step it was requested.
        """
        if self._aid_delivered:
            return False, "external aid already delivered"
        if self._aid_requested:
            return False, "external aid already requested"
        if (
            self._aid_deadline_hour is not None
            and simulation_hour >= self._aid_deadline_hour
        ):
            return False, (
                f"external aid deadline was hour {self._aid_deadline_hour}; "
                f"current hour {simulation_hour} — aid will NOT arrive"
            )
        self._aid_requested    = True
        self._aid_arrival_step = current_step + 1   # FIX 1
        return True, "external aid requested — arrives next step"

    def collect_external_aid(
        self,
        current_step: int,   # FIX 1: added parameter
    ) -> Optional[Dict[str, int]]:
        """
        FIX 1: Only delivers aid when current_step >= _aid_arrival_step,
        preventing same-step delivery.
        """
        if (
            self._aid_requested
            and not self._aid_delivered
            and self._aid_payload
            and self._aid_arrival_step is not None
            and current_step >= self._aid_arrival_step   # FIX 1
        ):
            self._aid_delivered = True
            return self._aid_payload
        return None

    # ── Road Upgrade ──────────────────────────────────────────────────────────

    @staticmethod
    def upgrade_road(current_status: RoadStatus) -> RoadStatus:
        """BLOCKED → DEGRADED → CLEAR → CLEAR (no-op)."""
        upgrade_map = {
            RoadStatus.BLOCKED:  RoadStatus.DEGRADED,
            RoadStatus.DEGRADED: RoadStatus.CLEAR,
            RoadStatus.CLEAR:    RoadStatus.CLEAR,
        }
        return upgrade_map[current_status]

    # ── Adjacency ─────────────────────────────────────────────────────────────

    @staticmethod
    def get_adjacent_zone_ids(
        zone_id:      int,
        all_zone_ids: List[int],
    ) -> List[int]:
        """Zones within ±2 IDs are considered neighbours."""
        return [z for z in all_zone_ids if 0 < abs(z - zone_id) <= 2]