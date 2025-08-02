# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Modified by LIU CHANG, 2025-06

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation

from M2oE.models.M2oE import M2oE


class M2oEActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_global_obs,
        num_actions,
        max_num_modules,
        m2oe_cfg: dict,
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        padding_mode: str = "learnable",
        padding_method: str = "concat",
        device: str = "cpu",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

        # learnable padding vector to pad observations to the maximum length
        if padding_mode == "learnable":
            self.padding_mode = padding_mode
            self.padding = nn.Parameter(torch.zeros(num_actor_obs))
        else:
            self.padding_mode = padding_mode
            self.padding = torch.zeros(num_actor_obs)

        self.padding_method = padding_method

        self.device = device

        # store observation dimension
        self.num_actor_obs = num_actor_obs

        hidden_dim = m2oe_cfg["hidden_dim"]
        activation = m2oe_cfg["activation"]

        self.actor = M2oE(
            num_obs=num_actor_obs,
            num_global_obs=num_global_obs,
            max_num_modules=max_num_modules,
            num_outputs=num_actions,
            hidden_dim=hidden_dim,
            num_experts=m2oe_cfg["num_experts"],
            activation=resolve_nn_activation(activation),
            gate_type=m2oe_cfg["gate_type"],
            gate_embedding_dim=m2oe_cfg["gate_embedding_dim"],
            gate_num_heads=m2oe_cfg["gate_num_heads"],
            gate_dropout=m2oe_cfg["gate_dropout"],
            device=self.device,
        )

        self.critic = M2oE(
            num_obs=num_actor_obs,
            num_global_obs=num_global_obs,
            max_num_modules=max_num_modules,
            num_outputs=1,
            hidden_dim=hidden_dim,
            num_experts=m2oe_cfg["num_experts"],
            activation=resolve_nn_activation(activation),
            gate_type=m2oe_cfg["gate_type"],
            gate_embedding_dim=m2oe_cfg["gate_embedding_dim"],
            gate_num_heads=m2oe_cfg["gate_num_heads"],
            gate_dropout=m2oe_cfg["gate_dropout"],
            device=self.device,
        )

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, obs_global):
        # compute mean
        mean = self.actor(observations, obs_global)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations, obs_global, **kwargs):
        self.update_distribution(observations, obs_global)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, obs_global):
        actions_mean = self.actor(observations, obs_global)
        return actions_mean

    def evaluate(self, critic_observations, obs_global, **kwargs):
        # critic_observations: [batch_size, num_obs_padded]
        # obs_global: [batch_size, num_global_obs]
        value = self.critic(critic_observations, obs_global)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True
