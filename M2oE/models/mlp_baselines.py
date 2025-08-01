import torch
import torch.nn as nn
from typing import Sequence

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
}


class JointMLPBaseline(nn.Module):
    """Baseline that predicts all actions jointly from concatenated observations."""

    def __init__(
        self,
        num_obs: int,
        num_global_obs: int,
        max_num_modules: int,
        num_actions: int,
        hidden_dims: Sequence[int],
        activation: str,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.device = device
        self.max_num_modules = max_num_modules
        self.modular_obs_dim = num_obs // max_num_modules
        self.modular_act_dim = num_actions // max_num_modules

        act_cls = _ACTIVATIONS.get(activation.lower())
        if act_cls is None:
            raise ValueError(f"Unsupported activation '{activation}'.")

        layers = []
        input_dim = num_obs + num_global_obs
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_cls())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_actions))
        self.mlp = nn.Sequential(*layers)

    def forward(self, modular_obs: torch.Tensor, global_obs: torch.Tensor) -> torch.Tensor:
        batch_size = modular_obs.shape[0]
        obs_flat = modular_obs.reshape(batch_size, -1)
        x = torch.cat([obs_flat, global_obs], dim=-1)
        return self.mlp(x)


class SharedModuleMLPBaseline(nn.Module):
    """Baseline with a shared MLP applied to each module separately."""

    def __init__(
        self,
        num_obs: int,
        num_global_obs: int,
        max_num_modules: int,
        num_actions: int,
        hidden_dims: Sequence[int],
        activation: str,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.device = device
        self.max_num_modules = max_num_modules
        self.modular_obs_dim = num_obs // max_num_modules
        self.modular_act_dim = num_actions // max_num_modules

        act_cls = _ACTIVATIONS.get(activation.lower())
        if act_cls is None:
            raise ValueError(f"Unsupported activation '{activation}'.")

        input_dim = self.modular_obs_dim + num_global_obs
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_cls())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, self.modular_act_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, modular_obs: torch.Tensor, global_obs: torch.Tensor) -> torch.Tensor:
        batch_size = modular_obs.shape[0]
        global_expanded = global_obs.unsqueeze(1).expand(-1, self.max_num_modules, -1)
        x = torch.cat([modular_obs, global_expanded], dim=-1)
        x = x.reshape(batch_size * self.max_num_modules, -1)
        actions = self.mlp(x)
        actions = actions.view(batch_size, self.max_num_modules, self.modular_act_dim)
        return actions.flatten(start_dim=1)
