# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import torch

from moonbot_envs.custom_lab_envs import CustomManagerBasedRLEnv
from rsl_rl.env import VecEnv
from M2oE.configs import morphology_configs


class ModulerRobotEnvWrapper(VecEnv):
    def __init__(self, env: CustomManagerBasedRLEnv, clip_actions: float | None = None):
        # check that input is valid
        if not isinstance(env.unwrapped, CustomManagerBasedRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv. Environment type:"
                f" {type(env)}"
            )
        # initialize the wrapper
        self.env = env
        self.clip_actions = clip_actions

        # store information required by wrapper
        self.num_morphologies = self.unwrapped.reward_manager.num_groups()
        # number of physical environments
        self.base_num_envs = self.unwrapped.num_envs
        # number of policy environments (all morphologies across envs)
        self.num_envs = self.base_num_envs * self.num_morphologies
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        self.num_actions = max(self.unwrapped.action_manager.group_action_dim.values())  # type: ignore
        obs_dict, extras = self.get_observations()
        morph_obs = {k: v for k, v in obs_dict.items() if "global" not in k}
        self.num_obs = max(
                [obs_tensor.shape[1] for obs_tensor in morph_obs.values()]
        )
        self.num_global_obs = obs_dict.get(f"global_obs_{morphology_configs.morphology_list[0]}", torch.zeros(1)).shape[1]
        self.num_act_sum = sum(self.unwrapped.action_manager.group_action_dim.values())  # type: ignore

        # -- privileged observations
        if (
            hasattr(self.unwrapped, "observation_manager")
            and "critic" in self.unwrapped.observation_manager.group_obs_dim
        ):
            self.num_privileged_obs = self.unwrapped.observation_manager.group_obs_dim["critic"][0]
        else:
            self.num_privileged_obs = 0

        # modify the action space to the clip range
        self._modify_action_space()

        self.env.reset()

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def cfg(self) -> object:
        """Returns the configuration class instance of the environment."""
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`observation_space`."""
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self) -> CustomManagerBasedRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped  # type: ignore

    """
    Properties
    """

    def get_observations(self):
        """Returns the current observations of the environment."""
        obs_dict = self.unwrapped.observation_manager.compute()
        return obs_dict, {"observations": obs_dict}

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):  # type: ignore
        self.unwrapped.episode_length_buf = value

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    def reset(self) -> tuple[torch.Tensor, dict]:  # noqa: D102
        # reset the environment
        obs_dict, _ = self.env.reset()
        # return observations
        return obs_dict, {"observations": obs_dict}  # type: ignore

    def step(self, actions: torch.Tensor):
        # process actions:
        # actions: [num_envs * num_morphologies , num_actions]
        actions = self._process_actions(actions)

        # clip actions
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # record step information
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        rew = self._process_rewards(rew)
        dones = (terminated | truncated).to(dtype=torch.long)
        dones = dones.unsqueeze(-1).repeat(1, self.num_morphologies).reshape(-1)
        # extras["time_outs"]:(num_envs, 1)
        extras["observations"] = obs_dict
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated.unsqueeze(-1).repeat(1, self.num_morphologies).reshape(-1, 1)

        # LIU CHANG: obs_dict needs to be processed in runner using learnable padding vector
        return obs_dict, rew, dones, extras

    def close(self):  # noqa: D102
        return self.env.close()

    """
    Helper functions
    """

    def _modify_action_space(self):
        """Modifies the action space to the clip range."""
        if self.clip_actions is None:
            return

        # modify the action space to the clip range
        # note: this is only possible for the box action space. we need to change it in the future for other action spaces.
        self.env.unwrapped.single_action_space = gym.spaces.Box(
            low=-self.clip_actions, high=self.clip_actions, shape=(self.num_actions,)
        )
        self.env.unwrapped.action_space = gym.vector.utils.batch_space(
            self.env.unwrapped.single_action_space, self.base_num_envs
        )

    def _process_actions(self, actions):
        """Convert flat action tensor into group dictionary for the environment."""

        # reshape into [base_num_envs, num_morphologies, num_actions]
        actions = actions.view(self.base_num_envs, self.num_morphologies, -1)

        group_dims = list(self.unwrapped.action_manager.group_action_dim.values())  # type: ignore
        group_names = list(self.unwrapped.action_manager.group_action_dim.keys())  # type: ignore

        processed_actions = {}
        for idx, (name, dim) in enumerate(zip(group_names, group_dims)):
            processed_actions[name] = actions[:, idx, :dim]

        return processed_actions

    def _process_rewards(self, rewards):
        reward_tensors = [rewards[key] for key in rewards.keys()]
        rewards = torch.stack(reward_tensors, dim=1)

        return rewards.view(self.num_envs)
