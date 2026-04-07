from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from app.models import (
    Action,
    AssessZone,
    CoordinateWithNGO,
    DispatchResource,
    EpisodeState,
    EstablishFieldHospital,
    Observation,
    OpenAlternativeRoute,
    PrioritizeZone,
    RequestExternalAid,
    ResourceInventory,
    RoadStatus,
    ZoneState,
)
from app.reward import RewardCalculator
from app.simulation import (
    HOURS_PER_STEP,
    MAX_STEPS,
    SimulationEngine,
    compute_survival_rate,
)
from app.tasks import TASK_REGISTRY
from app.tasks.base_task import BaseTask


class DisasterEnv:
    """
    Singleton RL environment. Owned by main.py as a module-level instance.
    One active episode at a time. No session IDs.
    """

    def __init__(self) -> None:
        self._engine:            Optional[SimulationEngine] = None
        self._task:              Optional[BaseTask]          = None
        self._task_id:           str                         = ""
        self._zones:             List[ZoneState]             = []
        self._total_inventory:   Optional[ResourceInventory] = None
        self._simulation_hour:   int                         = 0
        self._current_step:      int                         = 0
        self._cumulative_reward: float                       = 0.0
        self._done:              bool                        = False
        self._events_log:        List[str]                   = []
        self._calculator:        Optional[RewardCalculator]  = None
        self._initialized:       bool                        = False

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def reset(self, task_id: str, seed: int) -> Observation:
        """
        Initialise a fresh episode. Fully deterministic per (task_id, seed).
        """
        if task_id not in TASK_REGISTRY:
            raise ValueError(
                f"Unknown task_id: '{task_id}'. "
                f"Valid options: {list(TASK_REGISTRY.keys())}"
            )

        self._task_id = task_id
        self._task    = TASK_REGISTRY[task_id]()
        self._engine  = SimulationEngine(seed=seed)

        self._zones, self._total_inventory = self._task.build(self._engine, seed)

        self._simulation_hour   = 0
        self._current_step      = 0
        self._cumulative_reward = 0.0
        self._done              = False
        self._events_log        = [f"Episode started: task={task_id} seed={seed}"]

        self._calculator = RewardCalculator(task_id=task_id)
        self._calculator.reset()

        self._initialized = True
        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, float, float, bool, dict]:
        """
        Advance simulation by one step (6 simulated hours).

        Execution order:
          1.  Collect returned resources
          2.  Deliver external aid if ready
          3.  Apply resource wave → update total_inventory
          4.  Compute duplicate_assess BEFORE marking zone as assessed
          5.  Process the action
          6.  Apply step casualties
          7.  Check and apply aftershock
          8.  Increment simulation_hour and current_step
          9.  Check done, compute survival rate
          10. Compute reward
          11. Build and return observation
        """
        if not self._initialized:
            raise RuntimeError("No active episode. Call /reset first.")
        if self._done:
            raise RuntimeError("Episode is already done. Call /reset to start a new episode.")

        info:        dict          = {}
        step_events: List[str]     = []
        pre_zones                  = list(self._zones)

        # ── 1. Collect returned resources ──────────────────────────────────────
        freed = self._engine.collect_returned_resources(self._current_step)
        if freed:
            msg = f"RESOURCES RETURNED: {freed}"
            self._events_log.append(msg)
            step_events.append(msg)
            info["returned_resources"] = freed

        # ── 2. Deliver external aid if it has arrived ──────────────────────────
        aid = self._engine.collect_external_aid(self._current_step)
        if aid:
            self._total_inventory = ResourceInventory(
                ambulances      = self._total_inventory.ambulances      + aid.get("ambulances",      0),
                food_trucks     = self._total_inventory.food_trucks     + aid.get("food_trucks",     0),
                field_hospitals = self._total_inventory.field_hospitals + aid.get("field_hospitals", 0),
                rescue_teams    = self._total_inventory.rescue_teams    + aid.get("rescue_teams",    0),
            )
            msg = f"EXTERNAL AID DELIVERED: {aid}"
            self._events_log.append(msg)
            step_events.append(msg)
            info["external_aid_delivered"] = aid

        # ── 3. Apply resource wave for current simulation_hour ─────────────────
        wave = self._engine.get_wave_at_hour(self._simulation_hour)
        if wave:
            self._total_inventory = ResourceInventory(
                ambulances      = self._total_inventory.ambulances      + wave.resources.get("ambulances",      0),
                food_trucks     = self._total_inventory.food_trucks     + wave.resources.get("food_trucks",     0),
                field_hospitals = self._total_inventory.field_hospitals + wave.resources.get("field_hospitals", 0),
                rescue_teams    = self._total_inventory.rescue_teams    + wave.resources.get("rescue_teams",    0),
            )
            msg = f"RESOURCE WAVE arrived at hour {self._simulation_hour}: {wave.resources}"
            self._events_log.append(msg)
            step_events.append(msg)
            info["resource_wave"] = wave.resources

        # ── 4. Duplicate-assess check BEFORE marking ───────────────────────────
        duplicate_assess = False
        if isinstance(action, AssessZone):
            duplicate_assess = self._engine.is_assessed(action.zone_id)
            self._engine.mark_assessed(action.zone_id)

        # ── 5. Process action (pass duplicate_assess — FIX) ────────────────────
        action_msg = self._process_action(action, info, duplicate_assess)
        self._events_log.append(action_msg)
        step_events.append(action_msg)

        # ── 6. Apply step casualties ───────────────────────────────────────────
        self._zones, casualty_msgs = self._engine.apply_step_casualties(self._zones)
        self._events_log.extend(casualty_msgs)
        step_events.extend(casualty_msgs)

        # ── 7. Apply aftershock at current simulation_hour ─────────────────────
        aftershock = self._engine.get_aftershock_at_hour(self._simulation_hour)
        if aftershock:
            self._zones, shock_msgs = self._engine.apply_aftershock(aftershock, self._zones)
            self._events_log.extend(shock_msgs)
            step_events.extend(shock_msgs)
            info["aftershock_zones_blocked"] = shock_msgs

        # ── 8. Advance clocks ──────────────────────────────────────────────────
        self._simulation_hour += HOURS_PER_STEP
        self._current_step    += 1

        # ── 9. Check done and survival rate ────────────────────────────────────
        all_served       = all(z.served for z in self._zones)
        self._done       = (self._current_step >= MAX_STEPS) or all_served
        current_survival = compute_survival_rate(self._zones)
        baseline         = self._task.baseline_survival_rate

        # ── 10. Compute reward ─────────────────────────────────────────────────
        reward = self._calculator.compute_step_reward(
            pre_zones              = pre_zones,
            post_zones             = self._zones,
            action                 = action,
            simulation_hour        = self._simulation_hour,
            done                   = self._done,
            overflow_occurred      = self._engine.hospital_overflow_occurred(),
            current_survival_rate  = current_survival,
            baseline_survival_rate = baseline,
            duplicate_assess       = duplicate_assess,
        )

        self._cumulative_reward += reward

        if self._done:
            self._cumulative_reward = max(0.0, min(1.0, self._cumulative_reward))

        info["simulation_hour"]        = self._simulation_hour
        info["current_survival_rate"]  = round(current_survival, 4)
        info["baseline_survival_rate"] = baseline
        info["step_events"]            = step_events

        # ── 11. Build observation ──────────────────────────────────────────────
        obs = self._build_observation()
        return (
            obs,
            round(reward, 4),
            round(self._cumulative_reward, 4),
            self._done,
            info,
        )

    def get_state(self) -> EpisodeState:
        """Return full episode snapshot. Called by GET /state."""
        if not self._initialized:
            raise RuntimeError("No active episode. Call /reset first.")
        return EpisodeState(
            task_id                = self._task_id,
            simulation_hour        = self._simulation_hour,
            zones                  = list(self._zones),
            resources              = self._total_inventory,
            cumulative_reward      = round(self._cumulative_reward, 4),
            done                   = self._done,
            events_log             = list(self._events_log),
            current_survival_rate  = round(compute_survival_rate(self._zones), 4),
            baseline_survival_rate = self._task.baseline_survival_rate,
        )

    # ─────────────────────────────────────────────
    # Action processing
    # ─────────────────────────────────────────────

    def _process_action(
        self,
        action:           Action,
        info:             dict,
        duplicate_assess: bool = False,   # FIX: passed from step()
    ) -> str:
        if isinstance(action, AssessZone):
            return self._handle_assess_zone(action, duplicate_assess)

        if isinstance(action, DispatchResource):
            return self._handle_dispatch_resource(action, info)

        if isinstance(action, EstablishFieldHospital):
            return self._handle_establish_hospital(action, info)

        if isinstance(action, OpenAlternativeRoute):
            return self._handle_open_route(action)

        if isinstance(action, RequestExternalAid):
            return self._handle_request_aid(action, info)

        if isinstance(action, PrioritizeZone):
            return f"PrioritizeZone: zone {action.zone_id} → priority={action.priority}"

        if isinstance(action, CoordinateWithNGO):
            return f"CoordinateWithNGO: ngo={action.ngo_id} task={action.task}"

        return f"Unknown action type: {type(action).__name__}"

    def _handle_assess_zone(
        self,
        action:       AssessZone,
        is_duplicate: bool = False,   # FIX: no longer calls engine.is_assessed()
    ) -> str:
        zone = self._zone_by_id(action.zone_id)
        if zone is None:
            return f"AssessZone: zone {action.zone_id} not found"
        tag = "duplicate" if is_duplicate else "assessed"
        return (
            f"AssessZone: zone {action.zone_id} [{tag}] "
            f"dmg={zone.damage_level} road={zone.road_status} "
            f"needs={zone.needs} casualties={zone.casualties}"
        )

    def _handle_dispatch_resource(
        self,
        action: DispatchResource,
        info:   dict,
    ) -> str:
        ok, msg = self._engine.dispatch(
            resource_type   = action.resource_type,
            quantity        = action.quantity,
            zone_id         = action.zone_id,
            current_step    = self._current_step,
            total_inventory = self._total_inventory,
        )
        info["dispatch_success"] = ok
        info["dispatch_message"] = msg

        if ok:
            self._zones = self._apply_dispatch_to_zones(self._zones, action)

        return (
            f"DispatchResource: type={action.resource_type} qty={action.quantity} "
            f"zone={action.zone_id} → {msg}"
        )

    def _handle_establish_hospital(
        self,
        action: EstablishFieldHospital,
        info:   dict,
    ) -> str:
        zone = self._zone_by_id(action.zone_id)
        if zone is None:
            return f"EstablishFieldHospital: zone {action.zone_id} not found"
        if zone.road_status == RoadStatus.BLOCKED:
            return f"EstablishFieldHospital: zone {action.zone_id} road BLOCKED — cannot establish"
        if zone.hospital_present:
            return f"EstablishFieldHospital: zone {action.zone_id} already has a hospital"

        ok, msg = self._engine.dispatch(
            resource_type   = "field_hospitals",
            quantity        = 1,
            zone_id         = action.zone_id,
            current_step    = self._current_step,
            total_inventory = self._total_inventory,
        )
        if ok:
            self._engine.init_hospital(action.zone_id)
            self._zones = [
                z.model_copy(update={"hospital_present": True, "served": True})
                if z.zone_id == action.zone_id else z
                for z in self._zones
            ]
            info["hospital_established"] = action.zone_id

        return f"EstablishFieldHospital: zone {action.zone_id} → {'established' if ok else msg}"

    def _handle_open_route(self, action: OpenAlternativeRoute) -> str:
        updated: List[ZoneState] = []
        result_msg = f"OpenAlternativeRoute: zone {action.zone_id} not found"
        for z in self._zones:
            if z.zone_id == action.zone_id:
                new_status = SimulationEngine.upgrade_road(z.road_status)
                updated.append(z.model_copy(update={"road_status": new_status}))
                result_msg = (
                    f"OpenAlternativeRoute: zone {action.zone_id} "
                    f"{z.road_status} → {new_status}"
                )
            else:
                updated.append(z)
        self._zones = updated
        return result_msg

    def _handle_request_aid(
        self,
        action: RequestExternalAid,
        info:   dict,
    ) -> str:
        ok, msg = self._engine.request_external_aid(
            simulation_hour = self._simulation_hour,
            current_step    = self._current_step,
        )
        info["aid_requested"] = ok
        info["aid_message"]   = msg
        return f"RequestExternalAid: type={action.resource_type} → {msg}"

    # ─────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────

    def _build_observation(self) -> Observation:
        weather = self._engine.build_weather_state(self._simulation_hour, self._zones)
        return Observation(
            zones                  = list(self._zones),
            resources              = self._total_inventory,
            simulation_hour        = self._simulation_hour,
            weather                = weather,
            active_events          = list(self._events_log[-5:]),
            baseline_survival_rate = self._task.baseline_survival_rate,
            current_survival_rate  = round(compute_survival_rate(self._zones), 4),
        )

    def _zone_by_id(self, zone_id: int) -> Optional[ZoneState]:
        return next((z for z in self._zones if z.zone_id == zone_id), None)

    def _apply_dispatch_to_zones(
        self,
        zones:  List[ZoneState],
        action: DispatchResource,
    ) -> List[ZoneState]:
        """
        Mark zone as served when dispatched resource matches a zone need
        and the road is not BLOCKED.
        """
        resource_to_needs: Dict[str, List[str]] = {
            "ambulances":      ["MEDICAL"],
            "food_trucks":     ["FOOD"],
            "field_hospitals": ["SHELTER", "MEDICAL"],
            "rescue_teams":    ["RESCUE", "MEDICAL"],
        }
        relevant = resource_to_needs.get(action.resource_type, [])

        updated = []
        for z in zones:
            if (
                z.zone_id == action.zone_id
                and not z.served
                and z.road_status != RoadStatus.BLOCKED
                and any(need in relevant for need in z.needs)
            ):
                updated.append(z.model_copy(update={"served": True}))
            else:
                updated.append(z)
        return updated