"""Task-specialized M2oE allocation for the MoonBot experiment.

Task-specialized means one M2oE is shared by all three morphologies *within one
logical task*.  The allocation is therefore:

    locomotion       -> one M2oE (Minimal + Dragon + Tricycle)
    arm_reach        -> one M2oE (Minimal + Dragon + Tricycle)
    obstacle_avoiding-> one M2oE (Minimal + Dragon + Tricycle)

The integrated Isaac Lab environments expose the three morphologies as module
groups in one observation/action space.  Each task can therefore keep its own
observation layout, while the morphology modules inside that task use the same
policy and optimizer.  This file is intentionally separate from the original
``Integration_Locomotion_v1`` runner.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from M2oE.models.modules.actor_critic import M2oEActorCritic

# Related direction: Yang et al., NeurIPS 2020, task-specific routing over
# modules, and Cheng et al., IEEE RA-L 8(6), 2023, DOI 10.1109/LRA.2023.3271445.
# Here, one task policy is shared across the three morphology modules, which is
# the task-specialized condition being compared against morphology-specialized
# sharing.
Task = Literal["locomotion", "arm_reach", "obstacle_avoiding"]
Morphology = Literal["minimal", "dragon", "tricycle"]
TASKS: tuple[Task, ...] = ("locomotion", "arm_reach", "obstacle_avoiding")
MORPHOLOGIES: tuple[Morphology, ...] = ("minimal", "dragon", "tricycle")


@dataclass(frozen=True)
class ObservationSpec:
    """Observation/action dimensions for one integrated task environment."""

    num_obs: int
    num_global_obs: int
    num_actions: int
    max_num_modules: int = 1


class TaskSpecializedM2oE(nn.Module):
    """Three M2oE actor-critics, one shared across morphologies per task."""

    def __init__(
        self,
        specs: dict[Task, ObservationSpec] | dict[tuple[Task, Morphology], ObservationSpec],
        model_cfg: dict,
        policy_cfg: dict,
        device: str = "cpu",
    ):
        super().__init__()
        self.task_specs = self._normalize_specs(specs)
        self.policies = nn.ModuleDict()
        for task in TASKS:
            spec = self.task_specs[task]
            if spec.num_obs % spec.max_num_modules:
                raise ValueError(f"num_obs must divide max_num_modules for task {task}")
            if spec.num_actions % spec.max_num_modules:
                raise ValueError(f"num_actions must divide max_num_modules for task {task}")
            self.policies[task] = M2oEActorCritic(
                spec.num_obs,
                spec.num_global_obs,
                spec.num_actions,
                spec.max_num_modules,
                model_cfg=dict(model_cfg),
                device=device,
                **dict(policy_cfg),
            )

    @staticmethod
    def _normalize_specs(specs):
        """Accept the old tuple-keyed form but collapse it to one spec per task.

        The tuple-keyed compatibility path is useful when a caller still builds
        specs from nine environment descriptors.  It rejects morphology
        dimensions that cannot be consumed by one shared task policy.
        """
        if set(specs) == set(TASKS):
            normalized = dict(specs)
        else:
            normalized = {}
            for task in TASKS:
                candidates = [specs[(task, morphology)] for morphology in MORPHOLOGIES
                              if (task, morphology) in specs]
                if len(candidates) != len(MORPHOLOGIES):
                    raise ValueError(f"Missing morphology specs for task {task}")
                first = candidates[0]
                if any(candidate != first for candidate in candidates[1:]):
                    raise ValueError(
                        f"Task-specialized policy for {task} requires one common "
                        "integrated observation/action spec across morphologies"
                    )
                normalized[task] = first
        missing = set(TASKS) - set(normalized)
        if missing:
            raise ValueError(f"Missing task specs: {sorted(missing)}")
        return normalized

    def policy_for(self, task: Task, morphology: Morphology | None = None) -> M2oEActorCritic:
        """Return the task policy; ``morphology`` is accepted for old callers.

        The optional morphology argument is intentionally ignored.  All three
        morphologies in the selected task use this same network.
        """
        if task not in TASKS:
            raise KeyError(f"Unknown task {task!r}")
        return self.policies[task]

    def optimizers(self, learning_rate: float):
        """Return one optimizer per task, shared by its three morphology modules."""
        return {
            task: torch.optim.Adam(self.policies[task].parameters(), lr=learning_rate)
            for task in TASKS
        }
