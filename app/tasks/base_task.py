from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

from app.models import ResourceInventory, ZoneState
from app.simulation import SimulationEngine


class BaseTask(ABC):
    """
    Abstract base class for all disaster scenarios.

    Subclasses must set class-level attributes and implement build().
    build() is the only method environment.py calls on a task object.
    """

    task_id:                str   = ""
    baseline_survival_rate: float = 0.0
    max_steps:              int   = 12

    @abstractmethod
    def build(
        self,
        engine: SimulationEngine,
        seed:   int,
    ) -> Tuple[List[ZoneState], ResourceInventory]:
        """
        Configure the engine for this scenario and return the initial world state.

        Responsibilities:
          1. Call engine.configure_* methods (aftershocks, waves, weather,
             no-go zones, external aid) as needed for this task.
          2. Return (initial_zones, initial_resources).

        The engine is already seeded. All rng calls must go through engine.rng
        for determinism.

        Parameters
        ----------
        engine  Pre-seeded SimulationEngine instance.
        seed    The integer seed (available if the task needs it for
                any additional deterministic generation).

        Returns
        -------
        Tuple of (List[ZoneState], ResourceInventory) representing
        the world at simulation_hour = 0.
        """
        ...