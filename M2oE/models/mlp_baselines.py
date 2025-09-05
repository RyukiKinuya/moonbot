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
        device: torch.device,
        num_obs: int,
        num_global_obs: int,
        max_num_modules: int,
        num_outputs: int,
        hidden_dims: Sequence[int],
        activation: str,
        dropout: float | None = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.max_num_modules = max_num_modules
        self.modular_obs_dim = num_obs // max_num_modules
        # Determine per-module act dim for order restoration
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

        layers = []
        input_dim = num_obs + num_global_obs
        # Input normalization to stabilize scale across morphologies/modules
        self.input_ln = nn.LayerNorm(input_dim)
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_cls())
            if dropout is not None and dropout > 0.0:
                layers.append(nn.Dropout(p=float(dropout)))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_outputs))
        self.mlp = nn.Sequential(*layers)

    def forward(self, modular_obs: torch.Tensor, global_obs: torch.Tensor, module_mask=None) -> torch.Tensor:
        batch_size = modular_obs.shape[0]
        # reshape to [B, M, Dm]
        M = self.max_num_modules
        Dm = self.modular_obs_dim
        obs_view = modular_obs.view(batch_size, M, Dm)

        need_shuffle = self.training and (module_mask is not None)
        if need_shuffle:
            # Vectorized per-sample permutation of valid modules
            rand_scores = torch.rand(batch_size, M, device=self.device)
            positions = torch.arange(M, device=self.device).unsqueeze(0).expand(batch_size, -1)
            scores = torch.where(
                module_mask,
                rand_scores,
                1.0 + positions.to(rand_scores.dtype) / (M + 1.0),
            )
            idx = torch.argsort(scores, dim=1)
            inv_idx = torch.argsort(idx, dim=1)
            gather_idx = idx.unsqueeze(-1).expand(-1, -1, Dm)
            obs_view = obs_view.gather(1, gather_idx)

        # flatten back to [B, M*Dm]
        obs_flat = obs_view.reshape(batch_size, -1)
        x = torch.cat([obs_flat, global_obs], dim=-1)
        x = self.input_ln(x)
        y = self.mlp(x)

        # If outputs are per-module, restore original order
        if not self.aggregate and need_shuffle:
            y_view = y.view(batch_size, M, self.modular_act_dim)
            gather_idx_out = inv_idx.unsqueeze(-1).expand(-1, -1, self.modular_act_dim)
            y = y_view.gather(1, gather_idx_out).flatten(start_dim=1)
        return y


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
        dropout: float | None = None,
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
        # Input normalization per (module, global) pair
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_cls())
            if dropout is not None and dropout > 0.0:
                layers.append(nn.Dropout(p=float(dropout)))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, self.modular_act_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, modular_obs: torch.Tensor, global_obs: torch.Tensor, module_mask=None) -> torch.Tensor:
        """Compute outputs from modular and global observations.

        Args:
            modular_obs: Tensor of shape ``[batch_size, num_obs]`` containing all modular
                observations concatenated.
            global_obs: Tensor of shape ``[batch_size, num_global_obs]``.

        Returns:
            Tensor of shape ``[batch_size, num_outputs]``.
        """
        batch_size = modular_obs.shape[0]
        M = self.max_num_modules
        Dm = self.modular_obs_dim
        modular_obs = modular_obs.view(batch_size, M, Dm)

        need_shuffle = self.training and (module_mask is not None)
        if need_shuffle:
            rand_scores = torch.rand(batch_size, M, device=self.device)
            positions = torch.arange(M, device=self.device).unsqueeze(0).expand(batch_size, -1)
            scores = torch.where(
                module_mask,
                rand_scores,
                1.0 + positions.to(rand_scores.dtype) / (M + 1.0),
            )
            idx = torch.argsort(scores, dim=1)
            inv_idx = torch.argsort(idx, dim=1)
            gather_idx_mod = idx.unsqueeze(-1).expand(-1, -1, Dm)
            modular_obs = modular_obs.gather(1, gather_idx_mod)

        global_expanded = global_obs.unsqueeze(1).expand(-1, M, -1)
        x = torch.cat([modular_obs, global_expanded], dim=-1)
        x = x.view(batch_size * M, -1)
        actions = self.mlp(x)
        actions = actions.view(batch_size, M, self.modular_act_dim)

        if need_shuffle:
            gather_idx_out = inv_idx.unsqueeze(-1).expand(-1, -1, self.modular_act_dim)
            actions = actions.gather(1, gather_idx_out)

        if self.aggregate:
            return actions.mean(dim=1)
        return actions.flatten(start_dim=1)
