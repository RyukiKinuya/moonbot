from __future__ import annotations

import torch
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerBase, ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from .manager_term_cfg import RewardGroupCfg


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class RewardManager(ManagerBase):
    _env: ManagerBasedRLEnv
    """The environment instance."""

    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        self._group_reward_term_names = dict()
        self._group_reward_term_cfgs = dict()
        self._class_term_cfgs = list()
        self._term_names = list()

        super().__init__(cfg, env)
        self._episode_sums = dict()
        self._reward_buf = dict()
        self._step_reward = dict()
        for group_name, term_names in self._group_reward_term_names.items():
            self._episode_sums[group_name] = dict()
            self._reward_buf[group_name] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self._step_reward[group_name] = torch.zeros((self.num_envs, len(term_names)), dtype=torch.float, device=self.device)
            for term_name in term_names:
                self._episode_sums[group_name][term_name] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                self._term_names.append("/" + group_name + "/" + term_name)


    def __str__(self) -> str:
        num_terms = sum(len(term_names) for term_names in self._group_reward_term_names.values())
        msg = f"<RewardManager> contains {len(self._group_reward_term_names)} groups with {num_terms} terms.\n"

        # create table for term information
        table = PrettyTable()
        table.title = "Active Reward Terms"
        table.field_names = ["Index", "Group", "Term", "Weight"]
        # set alignment of table columns
        table.align["Group"] = "l"
        table.align["Term"] = "l"
        table.align["Weight"] = "r"
        # add info on each term
        idx = 0
        for group_name, term_names in self._group_reward_term_names.items():
            for term_name, term_cfg in zip(term_names, self._group_reward_term_cfgs[group_name]):
                table.add_row([idx, group_name, term_name, term_cfg.weight])
                idx += 1

        msg += table.get_string()
        msg += "\n"

        return msg

    """
    Properties.
    """

    @property
    def active_terms(self) -> list[str]:
        """Name of active reward terms."""
        return self._term_names

    def num_groups(self) -> int:
        """Returns the number of reward groups."""
        return len(self._group_reward_term_names)

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Returns the episodic sum of individual reward terms.

        Args:
            env_ids: The environment ids for which the episodic sum of
                individual reward terms is to be returned. Defaults to all the environment ids.

        Returns:
            Dictionary of episodic sum of individual reward terms.
        """
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        # store information
        extras = {}
        for group_name, term_names in self._group_reward_term_names.items():
            group_sum = 0.0
            for term_name in term_names:
                term_sum_tensor = self._episode_sums[group_name][term_name][env_ids]
                term_mean = term_sum_tensor.mean()
                key = f"Episode_Reward/{group_name}/{term_name}"
                extras[key] = term_mean / self._env.max_episode_length_s
                group_sum += term_mean

                self._episode_sums[group_name][term_name][env_ids] = 0.0

            for term_cfg in self._class_term_cfgs:
                if term_cfg.func.group_name == group_name:
                    term_cfg.func.reset(env_ids=env_ids)

        return extras

    def compute(self, dt: float):
        # reset _reward_buf
        self._reward_buf = dict()

        for group_name, term_names in self._group_reward_term_names.items():
            self._reward_buf[group_name] = self.compute_group(group_name, dt)

        return self._reward_buf

    def compute_group(self, group_name, dt: float):
        if group_name not in self._group_reward_term_names:
            raise ValueError(f"Reward group '{group_name}' not found.")

        group_reward = 0.0
        idx = 0

        for name, term_cfg in zip(self._group_reward_term_names[group_name], self._group_reward_term_cfgs[group_name]):

            if term_cfg.weight == 0.0:
                continue

            value = term_cfg.func.compute(self._env, **term_cfg.params) * term_cfg.weight * dt

            group_reward += value

            self._episode_sums[group_name][name] += value
            self._step_reward[group_name][:, idx] = value / dt
            idx = idx + 1

        return group_reward


    def set_term_cfg(self, group_name: str, term_name: str, cfg: RewardTermCfg):
        if group_name not in self._group_reward_term_names:
            raise ValueError(f"Reward Group '{group_name}' not found.")

        if term_name not in self._group_reward_term_names[group_name]:
            raise ValueError(f"Reward Term '{term_name}' not found in group '{group_name}'.")

        idx = self._group_reward_term_names[group_name].index(term_name)
        self._group_reward_term_cfgs[group_name][idx] = cfg

    def get_term_cfg(self, group_name: str, term_name: str) -> RewardTermCfg:
        if group_name not in self._group_reward_term_names:
            raise ValueError(f"Reward Group '{group_name}' not found.")
        if term_name not in self._group_reward_term_names[group_name]:
            raise ValueError(f"Reward Term '{term_name}' not found in group '{group_name}'.")

        idx = self._group_reward_term_names[group_name].index(term_name)
        return self._group_reward_term_cfgs[group_name][idx]

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        terms = []
        for group_name, term_names in self._group_reward_term_names.items():
            for i, term_name in enumerate(term_names):
                value = self._step_reward[group_name][env_idx, i].cpu().item()
                terms.append((f"{group_name}/{term_name}", [value]))
        return terms

    """
    Helper functions.
    """

    def _prepare_terms(self):
        if isinstance(self.cfg, dict):
            group_cfg_items = self.cfg.items()
        else:
            group_cfg_items = self.cfg.__dict__.items()
        # iterate over all the terms
        for group_name, group_cfg in group_cfg_items:
            if group_cfg is None:
                continue

            if isinstance(group_cfg, dict):
                cfg_items = group_cfg.items()
            else:
                cfg_items = group_cfg.__dict__.items()

            if not isinstance(group_cfg, RewardGroupCfg):
                raise TypeError(
                    f"Configuration for the group '{group_name}' is not of type RewardGroupCfg."
                    f" Received: '{type(group_cfg)}'."
                )

            self._group_reward_term_names[group_name] = list()
            self._group_reward_term_cfgs[group_name] = list()

            for term_name, term_cfg in cfg_items:
                # check for non config
                if term_cfg is None:
                    continue
                # check for valid config type
                if not isinstance(term_cfg, RewardTermCfg):
                    raise TypeError(
                        f"Configuration for the term '{term_name}' is not of type RewardTermCfg."
                        f" Received: '{type(term_cfg)}'."
                    )
                # check for valid weight type
                if not isinstance(term_cfg.weight, (float, int)):
                    raise TypeError(
                        f"Weight for the term '{term_name}' is not of type float or int."
                        f" Received: '{type(term_cfg.weight)}'."
                    )
                # resolve common parameters
                self._resolve_common_term_cfg(term_name, term_cfg, min_argc=1)
                # add function to list
                self._group_reward_term_names[group_name].append(term_name)
                self._group_reward_term_cfgs[group_name].append(term_cfg)
                # check if the term is a class
                if isinstance(term_cfg.func, ManagerTermBase):
                    self._class_term_cfgs.append(term_cfg)
