"""Task-specialized allocation for the nine MoonBot task/morphology streams.

Create one independent M2oE actor-critic for each of:
  locomotion x {minimal, dragon, tricycle}
  arm_reach x {minimal, dragon, tricycle}
  obstacle_avoiding x {minimal, dragon, tricycle}

Each network is updated only from its own stream. Raw observations can therefore
retain task-specific layouts. A rollout coordinator must provide the stream's
ObservationSpec and call ``policy_for(task, morphology)`` during collection.
This file is intentionally separate from Integration_Locomotion_v1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from M2oE.models.modules.actor_critic import M2oEActorCritic

# Related direction: Yang et al., NeurIPS 2020, task-specific routing over modules,
# https://proceedings.neurips.cc/paper_files/paper/2020/hash/32cfdce9631d8c7906e8e9d6e68b514b-Abstract.html
# and Cheng et al., IEEE RA-L 8(6), 2023, DOI 10.1109/LRA.2023.3271445.
# This implementation uses independent models as a stricter task-specialization baseline.
Task = Literal["locomotion", "arm_reach", "obstacle_avoiding"]
Morphology = Literal["minimal", "dragon", "tricycle"]
TASKS: tuple[Task, ...] = ("locomotion", "arm_reach", "obstacle_avoiding")
MORPHOLOGIES: tuple[Morphology, ...] = ("minimal", "dragon", "tricycle")
STREAMS = tuple((task, morphology) for task in TASKS for morphology in MORPHOLOGIES)


@dataclass(frozen=True)
class ObservationSpec:
    num_obs: int
    num_global_obs: int
    num_actions: int
    max_num_modules: int = 1


class TaskSpecializedM2oE(nn.Module):
    """Nine independent MoE actor-critics, one for each experiment stream."""

    def __init__(self, specs: dict[tuple[Task, Morphology], ObservationSpec],
                 model_cfg: dict, policy_cfg: dict, device: str = "cpu"):
        super().__init__()
        missing = set(STREAMS) - set(specs)
        if missing:
            raise ValueError(f"Missing stream specs: {sorted(missing)}")
        self.policies = nn.ModuleDict()
        self.stream_specs = dict(specs)
        for task, morphology in STREAMS:
            spec = specs[(task, morphology)]
            if spec.num_obs % spec.max_num_modules:
                raise ValueError(f"num_obs must divide max_num_modules for {task}/{morphology}")
            if spec.num_actions % spec.max_num_modules:
                raise ValueError(f"num_actions must divide max_num_modules for {task}/{morphology}")
            self.policies[self._key(task, morphology)] = M2oEActorCritic(
                spec.num_obs, spec.num_global_obs, spec.num_actions, spec.max_num_modules,
                model_cfg=dict(model_cfg), device=device, **dict(policy_cfg),
            )

    @staticmethod
    def _key(task: str, morphology: str) -> str:
        return f"{task}__{morphology}"

    def policy_for(self, task: Task, morphology: Morphology) -> M2oEActorCritic:
        return self.policies[self._key(task, morphology)]

    def optimizers(self, learning_rate: float):
        """Return separate optimizers so no gradients cross task boundaries."""
        return {key: torch.optim.Adam(policy.parameters(), lr=learning_rate)
                for key, policy in self.policies.items()}
