import torch
import torch.nn as nn


class TransformerBaseline(nn.Module):
    """Transformer-based baseline that maps modular and global observations to actions."""

    def __init__(
        self,
        num_obs: int,
        num_global_obs: int,
        max_num_modules: int,
        num_actions: int,
        embedding_dim: int,

        num_heads: int,
        dropout: float,
        device: torch.device,
        activation: str = "relu",
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.device = device
        self.max_num_modules = max_num_modules
        self.modular_obs_dim = num_obs // max_num_modules
        self.modular_act_dim = num_actions // max_num_modules

        self.input_projection_modular = nn.Linear(self.modular_obs_dim, embedding_dim)
        self.input_projection_global = nn.Linear(num_global_obs, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.action_head = nn.Linear(embedding_dim, self.modular_act_dim)

    def forward(self, modular_obs: torch.Tensor, global_obs: torch.Tensor) -> torch.Tensor:
        """Compute actions from modular and global observations.

        Args:
            modular_obs: Tensor of shape ``[batch_size, max_num_modules, modular_obs_dim]``.
            global_obs: Tensor of shape ``[batch_size, num_global_obs]``.

        Returns:
            Tensor of shape ``[batch_size, num_actions]`` containing actions for all modules.
        """
        feature_modular = self.input_projection_modular(modular_obs)
        feature_global = self.input_projection_global(global_obs).unsqueeze(1)
        tokens = torch.cat([feature_global, feature_modular], dim=1)

        encoded = self.transformer(tokens)
        action_tokens = encoded[:, 1:, :]
        actions = self.action_head(action_tokens)
        actions = actions.flatten(start_dim=1)
        return actions
