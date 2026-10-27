"""Morphology-specialized allocation for the nine MoonBot task streams.

Create one M2oE actor-critic for each morphology. Its weights are shared across
locomotion, arm reach and obstacle avoiding. Since their raw observations can
differ, each task has a trainable observation projection into that morphology's
common M2oE input schema. Per-module dimensions and action dimensions must be
consistent across the three tasks for a morphology.

A rollout coordinator must use ``adapt_observations`` before policy act/evaluate
and include each task stream in the same PPO update for that morphology. This
experiment is separate from Integration_Locomotion_v1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from M2oE.models.modules.actor_critic import M2oEActorCritic

# Related direction: Xiong et al., ICML 2023, morphology-conditioned control,
# https://proceedings.mlr.press/v202/xiong23a.html; Luo et al., IJCAI 2025,
# https://www.ijcai.org/proceedings/2025/0972.pdf. This is an experimental baseline,
# not a reproduction of either paper.
Task = Literal["locomotion", "arm_reach", "obstacle_avoiding"]
Morphology = Literal["minimal", "dragon", "tricycle"]
TASKS: tuple[Task, ...] = ("locomotion", "arm_reach", "obstacle_avoiding")
MORPHOLOGIES: tuple[Morphology, ...] = ("minimal", "dragon", "tricycle")


@dataclass(frozen=True)
class ObservationSpec:
    num_obs: int
    num_global_obs: int
    num_actions: int
    max_num_modules: int = 1


class MorphologySpecializedM2oE(nn.Module):
    """Three MoEs; each morphology network learns all three of its task streams."""

    def __init__(self, specs: dict[tuple[Task, Morphology], ObservationSpec],
                 model_cfg: dict, policy_cfg: dict, device: str = "cpu"):
        super().__init__()
        missing = {(t, m) for t in TASKS for m in MORPHOLOGIES} - set(specs)
        if missing:
            raise ValueError(f"Missing stream specs: {sorted(missing)}")
        self.policies = nn.ModuleDict()
        self.obs_adapters = nn.ModuleDict()
        self.global_adapters = nn.ModuleDict()
        self.stream_specs = dict(specs)
        self.common_specs: dict[Morphology, ObservationSpec] = {}

        for morphology in MORPHOLOGIES:
            rows = [specs[(task, morphology)] for task in TASKS]
            actions, modules = rows[0].num_actions, rows[0].max_num_modules
            if any(s.num_actions != actions or s.max_num_modules != modules for s in rows):
                raise ValueError(f"Action and module dimensions must match across tasks for {morphology}")
            common = ObservationSpec(
                num_obs=max(s.num_obs for s in rows),
                num_global_obs=max(s.num_global_obs for s in rows),
                num_actions=actions,
                max_num_modules=modules,
            )
            if common.num_obs % modules:
                raise ValueError(f"Unified observation size {common.num_obs} must divide module count {modules}")
            self.common_specs[morphology] = common
            self.policies[morphology] = M2oEActorCritic(
                common.num_obs, common.num_global_obs, common.num_actions, common.max_num_modules,
                model_cfg=dict(model_cfg), device=device, **dict(policy_cfg),
            )
            for task in TASKS:
                spec = specs[(task, morphology)]
                key = self._key(task, morphology)
                self.obs_adapters[key] = nn.Linear(spec.num_obs, common.num_obs)
                self.global_adapters[key] = nn.Linear(spec.num_global_obs, common.num_global_obs)

    @staticmethod
    def _key(task: str, morphology: str) -> str:
        return f"{task}__{morphology}"

    def policy_for(self, task: Task, morphology: Morphology) -> M2oEActorCritic:
        # Task is accepted to keep the same stream API as TaskSpecializedM2oE.
        if task not in TASKS:
            raise KeyError(task)
        return self.policies[morphology]

    def adapt_observations(self, task: Task, morphology: Morphology,
                           obs: torch.Tensor, global_obs: torch.Tensor):
        """Map task-specific states to the morphology's shared input schema."""
        key = self._key(task, morphology)
        return self.policies[morphology], self.obs_adapters[key](obs), self.global_adapters[key](global_obs)

    def optimizers(self, learning_rate: float):
        """One optimizer per morphology, including all three task input adapters."""
        result = {}
        for morphology in MORPHOLOGIES:
            params = list(self.policies[morphology].parameters())
            for task in TASKS:
                key = self._key(task, morphology)
                params.extend(self.obs_adapters[key].parameters())
                params.extend(self.global_adapters[key].parameters())
            result[morphology] = torch.optim.Adam(params, lr=learning_rate)
        return result
