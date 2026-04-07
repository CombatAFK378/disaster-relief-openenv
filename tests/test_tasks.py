from app.models import ResourceInventory, RoadStatus
from app.simulation import SimulationEngine
from app.tasks import TASK_REGISTRY, FloodEasyTask, EarthquakeMediumTask, CompoundHardTask
from app.tasks.compound_hard import NO_GO_ZONE_IDS


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def make_engine(seed: int = 42) -> SimulationEngine:
    return SimulationEngine(seed=seed)


# ─────────────────────────────────────────────
# Test 1 — all tasks build without error
# ─────────────────────────────────────────────

def test_all_tasks_load():
    for task_cls in [FloodEasyTask, EarthquakeMediumTask, CompoundHardTask]:
        task   = task_cls()
        engine = make_engine()
        result = task.build(engine, seed=42)

        assert isinstance(result, tuple), \
            f"{task_cls.__name__}: build() must return a tuple"
        assert len(result) == 2, \
            f"{task_cls.__name__}: build() must return (zones, resources)"
        zones, resources = result
        assert isinstance(zones, list), \
            f"{task_cls.__name__}: first element must be a list"
        assert isinstance(resources, ResourceInventory), \
            f"{task_cls.__name__}: second element must be ResourceInventory"
        assert len(zones) > 0, \
            f"{task_cls.__name__}: zones list must not be empty"


# ─────────────────────────────────────────────
# Test 2 — flood_easy zone count
# ─────────────────────────────────────────────

def test_flood_easy_zone_count():
    task     = FloodEasyTask()
    zones, _ = task.build(make_engine(), seed=42)
    assert len(zones) == 8, f"Expected 8 zones, got {len(zones)}"


# ─────────────────────────────────────────────
# Test 3 — earthquake_medium zone count
# ─────────────────────────────────────────────

def test_earthquake_medium_zone_count():
    task     = EarthquakeMediumTask()
    zones, _ = task.build(make_engine(), seed=42)
    assert len(zones) == 15, f"Expected 15 zones, got {len(zones)}"


# ─────────────────────────────────────────────
# Test 4 — compound_hard zone count
# ─────────────────────────────────────────────

def test_compound_hard_zone_count():
    task     = CompoundHardTask()
    zones, _ = task.build(make_engine(), seed=42)
    assert len(zones) == 20, f"Expected 20 zones, got {len(zones)}"


# ─────────────────────────────────────────────
# Test 5 — compound_hard NO-GO zones configured correctly
# ─────────────────────────────────────────────

def test_compound_hard_no_go_zones():
    task   = CompoundHardTask()
    engine = make_engine()
    task.build(engine, seed=42)

    assert engine._no_go_zones == set(NO_GO_ZONE_IDS), (
        f"Expected NO-GO zones {set(NO_GO_ZONE_IDS)}, got {engine._no_go_zones}"
    )


# ─────────────────────────────────────────────
# Test 6 — baseline survival rates
# ─────────────────────────────────────────────

def test_baseline_survival_rates():
    assert FloodEasyTask.baseline_survival_rate        == 0.61
    assert EarthquakeMediumTask.baseline_survival_rate == 0.54
    assert CompoundHardTask.baseline_survival_rate     == 0.48


# ─────────────────────────────────────────────
# Test 7 — flood_easy has exactly 1 blocked road
# ─────────────────────────────────────────────

def test_flood_easy_blocked_road_count():
    task     = FloodEasyTask()
    zones, _ = task.build(make_engine(), seed=42)
    blocked  = [z for z in zones if z.road_status == RoadStatus.BLOCKED]
    assert len(blocked) == 1, f"Expected 1 blocked zone, got {len(blocked)}"


# ─────────────────────────────────────────────
# Test 8 — earthquake_medium has exactly 6 blocked roads
# ─────────────────────────────────────────────

def test_earthquake_medium_blocked_road_count():
    task     = EarthquakeMediumTask()
    zones, _ = task.build(make_engine(), seed=42)
    blocked  = [z for z in zones if z.road_status == RoadStatus.BLOCKED]
    assert len(blocked) == 6, f"Expected 6 blocked zones, got {len(blocked)}"


# ─────────────────────────────────────────────
# Test 9 — compound_hard has exactly 12 blocked roads
# ─────────────────────────────────────────────

def test_compound_hard_blocked_road_count():
    task     = CompoundHardTask()
    zones, _ = task.build(make_engine(), seed=42)
    blocked  = [z for z in zones if z.road_status == RoadStatus.BLOCKED]
    assert len(blocked) == 12, f"Expected 12 blocked zones, got {len(blocked)}"


# ─────────────────────────────────────────────
# Test 10 — task registry has exactly the 3 expected keys
# ─────────────────────────────────────────────

def test_task_registry_keys():
    expected = {"flood_easy", "earthquake_medium", "compound_hard"}
    assert set(TASK_REGISTRY.keys()) == expected, (
        f"Expected registry keys {expected}, got {set(TASK_REGISTRY.keys())}"
    )