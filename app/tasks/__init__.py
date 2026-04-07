from app.tasks.flood_easy import FloodEasyTask
from app.tasks.earthquake_medium import EarthquakeMediumTask
from app.tasks.compound_hard import CompoundHardTask

TASK_REGISTRY = {
    "flood_easy":         FloodEasyTask,
    "earthquake_medium":  EarthquakeMediumTask,
    "compound_hard":      CompoundHardTask,
}

__all__ = [
    "FloodEasyTask",
    "EarthquakeMediumTask",
    "CompoundHardTask",
    "TASK_REGISTRY",
]