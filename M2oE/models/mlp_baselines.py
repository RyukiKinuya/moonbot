import torch
import torch.nn as nn
from typing import Sequence

_ACTIVATIONS = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
}


class JointMLPBaseline(nn.Module):
    """Baseline that predicts outputs jointly from concatenated observations."""

    def __init__(
        self,
        num_obs: int,
        num_global_obs: int,
        max_num_modules: int,
        num_outputs: int,
        hidden_dims: Sequence[int],
        activation: str,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.device = device
        self.max_num_modules = max_num_modules

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
        layers.append(nn.Linear(prev_dim, num_outputs))
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
        num_outputs: int,
        hidden_dims: Sequence[int],
        activation: str,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.device = device
        self.max_num_modules = max_num_modules
        self.modular_obs_dim = num_obs // max_num_modules
        if num_outputs == 1:
            self.modular_act_dim = 1
            self.aggregate = True
        else:
            if num_outputs % max_num_modules != 0:
                raise ValueError("num_outputs must be divisible by max_num_modules")
            self.modular_act_dim = num_outputs // max_num_modules
            self.aggregate = False

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
        """Compute outputs from modular and global observations.

        Args:
            modular_obs: Tensor of shape ``[batch_size, num_obs]`` containing all modular
                observations concatenated.
            global_obs: Tensor of shape ``[batch_size, num_global_obs]``.

        Returns:
            Tensor of shape ``[batch_size, num_outputs]``.
        """
        batch_size = modular_obs.shape[0]
        modular_obs = modular_obs.view(batch_size, self.max_num_modules, self.modular_obs_dim)
        global_expanded = global_obs.unsqueeze(1).expand(-1, self.max_num_modules, -1)
        x = torch.cat([modular_obs, global_expanded], dim=-1)
        x = x.view(batch_size * self.max_num_modules, -1)
        actions = self.mlp(x)
        actions = actions.view(batch_size, self.max_num_modules, self.modular_act_dim)
        if self.aggregate:
            return actions.mean(dim=1)
        return actions.flatten(start_dim=1)
