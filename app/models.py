from __future__ import annotations

from enum import Enum
from typing import Annotated, Dict, List, Literal, Union

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────

class DamageLevel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MODERATE = "MODERATE"
    LOW      = "LOW"


class RoadStatus(str, Enum):
    CLEAR    = "CLEAR"
    DEGRADED = "DEGRADED"
    BLOCKED  = "BLOCKED"


class NeedType(str, Enum):
    MEDICAL = "MEDICAL"
    FOOD    = "FOOD"
    SHELTER = "SHELTER"
    RESCUE  = "RESCUE"


class WeatherCondition(str, Enum):
    CLEAR = "CLEAR"
    RAIN  = "RAIN"
    STORM = "STORM"


class ResourceType(str, Enum):
    AMBULANCES      = "ambulances"
    FOOD_TRUCKS     = "food_trucks"
    FIELD_HOSPITALS = "field_hospitals"
    RESCUE_TEAMS    = "rescue_teams"


class ZonePriority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"


# ─────────────────────────────────────────────
# Core State Models
# ─────────────────────────────────────────────

class ZoneState(BaseModel):
    zone_id:          int
    population:       int
    damage_level:     DamageLevel
    road_status:      RoadStatus
    needs:            List[NeedType]
    casualties:       int
    served:           bool
    hospital_present: bool

    model_config = {"use_enum_values": True}


class ResourceInventory(BaseModel):
    ambulances:      int = Field(..., ge=0)
    food_trucks:     int = Field(..., ge=0)
    field_hospitals: int = Field(..., ge=0)
    rescue_teams:    int = Field(..., ge=0)


class WeatherState(BaseModel):
    condition:     WeatherCondition
    affects_zones: List[int]

    model_config = {"use_enum_values": True}


class Observation(BaseModel):
    zones:                  List[ZoneState]
    resources:              ResourceInventory
    simulation_hour:        int
    weather:                WeatherState
    active_events:          List[str]
    baseline_survival_rate: float
    current_survival_rate:  float


# ─────────────────────────────────────────────
# Action Models — discriminated union on `type`
# ─────────────────────────────────────────────

class AssessZone(BaseModel):
    type:    Literal["AssessZone"] = "AssessZone"
    zone_id: int


class DispatchResource(BaseModel):
    type:          Literal["DispatchResource"] = "DispatchResource"
    resource_type: ResourceType
    quantity:      int = Field(..., ge=1)
    zone_id:       int

    model_config = {"use_enum_values": True}


class EstablishFieldHospital(BaseModel):
    type:    Literal["EstablishFieldHospital"] = "EstablishFieldHospital"
    zone_id: int


class OpenAlternativeRoute(BaseModel):
    type:    Literal["OpenAlternativeRoute"] = "OpenAlternativeRoute"
    zone_id: int


class RequestExternalAid(BaseModel):
    type:          Literal["RequestExternalAid"] = "RequestExternalAid"
    resource_type: ResourceType

    model_config = {"use_enum_values": True}


class PrioritizeZone(BaseModel):
    type:     Literal["PrioritizeZone"] = "PrioritizeZone"
    zone_id:  int
    priority: ZonePriority

    model_config = {"use_enum_values": True}


class CoordinateWithNGO(BaseModel):
    type:   Literal["CoordinateWithNGO"] = "CoordinateWithNGO"
    ngo_id: str
    task:   str


# FIX 1: No StepRequest wrapper — Action used directly as endpoint body
Action = Annotated[
    Union[
        AssessZone,
        DispatchResource,
        EstablishFieldHospital,
        OpenAlternativeRoute,
        RequestExternalAid,
        PrioritizeZone,
        CoordinateWithNGO,
    ],
    Field(discriminator="type"),
]


# ─────────────────────────────────────────────
# API Envelope Models
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "flood_easy"
    seed:    int = 42


class StepResult(BaseModel):
    observation:       Observation
    reward:            float
    cumulative_reward: float
    done:              bool
    info:              Dict


# FIX 2: current_survival_rate and baseline_survival_rate added to EpisodeState
class EpisodeState(BaseModel):
    task_id:                str
    simulation_hour:        int
    zones:                  List[ZoneState]
    resources:              ResourceInventory
    cumulative_reward:      float
    done:                   bool
    events_log:             List[str]
    current_survival_rate:  float
    baseline_survival_rate: float


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"