from __future__ import annotations

import torch
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerBase
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg

from .manager_term_cfg import ActionGroupCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class GroupActionManager(ManagerBase):
    _env: ManagerBasedEnv

    def __init__(self, cfg: object, env: ManagerBasedEnv):
        self._group_action_term_names = dict()
        self._group_action_term_cfgs = dict()
        self._group_action_terms = dict()
        self._group_action_term_dim = dict()
        self._term_names = list()
        super().__init__(cfg, env)
        # buffers for actions
        self._action = {}
        self._prev_action = {}
        for group_name, dims in self._group_action_term_dim.items():
            dim = sum(dims)
            self._action[group_name] = torch.zeros((self.num_envs, dim), device=self.device)
            self._prev_action[group_name] = torch.zeros_like(self._action[group_name])

    def __str__(self) -> str:
        num_terms = sum(len(n) for n in self._group_action_term_names.values())
        msg = f"<GroupActionManager> contains {len(self._group_action_term_names)} groups with {num_terms} terms.\n"
        table = PrettyTable()
        table.title = "Active Action Terms"
        table.field_names = ["Index", "Group", "Term", "Dim"]
        table.align["Group"] = "l"
        table.align["Term"] = "l"
        table.align["Dim"] = "r"
        idx = 0
        for group_name, term_names in self._group_action_term_names.items():
            for term_name, term in zip(term_names, self._group_action_terms[group_name]):
                table.add_row([idx, group_name, term_name, term.action_dim])
                idx += 1
        msg += table.get_string()
        msg += "\n"
        return msg

    @property
    def group_action_dim(self) -> dict[str, int]:
        return {g: sum(self._group_action_term_dim[g]) for g in self._group_action_term_dim}

    @property
    def total_action_dim(self) -> int:
        return sum(self.group_action_dim.values())

    @property
    def active_terms(self) -> dict[str, list[str]]:
        """Name of active action terms for each group."""
        return self._group_action_term_names

    @property
    def action(self) -> dict[str, torch.Tensor]:
        return self._action

    @property
    def prev_action(self) -> dict[str, torch.Tensor]:
        return self._prev_action

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        for group_name in self._group_action_term_names:
            self._action[group_name][env_ids] = 0.0
            self._prev_action[group_name][env_ids] = 0.0
            for term in self._group_action_terms[group_name]:
                term.reset(env_ids=env_ids)
        return {}

    def process_action(self, actions: dict[str, torch.Tensor]):
        for group_name, group_actions in actions.items():
            if group_name not in self._group_action_term_names:
                raise ValueError(
                    f"Action group '{group_name}' not found. Available groups: {list(self._group_action_term_names.keys())}."
                )
            expected_dim = self.group_action_dim[group_name]
            if group_actions.shape[1] != expected_dim:
                raise ValueError(
                    f"Invalid action shape for group '{group_name}', expected {expected_dim}, got {group_actions.shape[1]}."
                )
            self._prev_action[group_name][:] = self._action[group_name]
            self._action[group_name][:] = group_actions.to(self.device)

            idx = 0
            for term in self._group_action_terms[group_name]:
                term_actions = group_actions[:, idx : idx + term.action_dim]
                term.process_actions(term_actions)
                idx += term.action_dim

    def apply_action(self) -> None:
        for group_name in self._group_action_terms:
            for term in self._group_action_terms[group_name]:
                term.apply_actions()

    def get_term(self, group_name: str, term_name: str) -> ActionTerm:
        if group_name not in self._group_action_term_names:
            raise ValueError(f"Action group '{group_name}' not found.")
        if term_name not in self._group_action_term_names[group_name]:
            raise ValueError(f"Action term '{term_name}' not found in group '{group_name}'.")
        idx = self._group_action_term_names[group_name].index(term_name)
        return self._group_action_terms[group_name][idx]

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        terms = []
        for group_name, term_names in self._group_action_term_names.items():
            idx = 0
            for term_name, term in zip(term_names, self._group_action_terms[group_name]):
                term_actions = self._action[group_name][env_idx, idx : idx + term.action_dim].cpu()
                terms.append((f"{group_name}/{term_name}", term_actions.tolist()))
                idx += term.action_dim
        return terms

    #
    # Helper functions.
    #
    def _prepare_terms(self):
        self._group_action_term_names = {}
        self._group_action_term_cfgs = {}
        self._group_action_terms = {}
        self._group_action_term_dim = {}
        if isinstance(self.cfg, dict):
            group_cfg_items = self.cfg.items()
        else:
            group_cfg_items = self.cfg.__dict__.items()
        for group_name, group_cfg in group_cfg_items:
            if group_cfg is None:
                continue
            if not isinstance(group_cfg, ActionGroupCfg):
                raise TypeError(
                    f"Configuration for the group '{group_name}' is not of type ActionGroupCfg. Received: '{type(group_cfg)}'."
                )
            if isinstance(group_cfg, dict):
                cfg_items = group_cfg.items()
            else:
                cfg_items = group_cfg.__dict__.items()
            self._group_action_term_names[group_name] = []
            self._group_action_term_cfgs[group_name] = []
            self._group_action_terms[group_name] = []
            self._group_action_term_dim[group_name] = []
            for term_name, term_cfg in cfg_items:
                if term_cfg is None:
                    continue
                if not isinstance(term_cfg, ActionTermCfg):
                    raise TypeError(
                        f"Configuration for the term '{term_name}' is not of type ActionTermCfg. Received: '{type(term_cfg)}'."
                    )
                term = term_cfg.class_type(term_cfg, self._env)
                if not isinstance(term, ActionTerm):
                    raise TypeError(
                        f"Returned object for the term '{term_name}' is not of type ActionTerm."
                    )
                self._group_action_term_names[group_name].append(term_name)
                self._group_action_term_cfgs[group_name].append(term_cfg)
                self._group_action_terms[group_name].append(term)
                self._group_action_term_dim[group_name].append(term.action_dim)
                self._term_names.append(f"{group_name}/{term_name}")
