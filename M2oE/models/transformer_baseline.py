import torch
import torch.nn as nn


class TransformerBaseline(nn.Module):
    """Transformer-based baseline that maps modular and global observations to outputs."""

    def __init__(
        self,
        num_obs: int,
        num_global_obs: int,
        max_num_modules: int,
        num_outputs: int,
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
        if num_outputs == 1:
            self.modular_act_dim = 1
            self.aggregate = True
        else:
            if num_outputs % max_num_modules != 0:
                raise ValueError("num_outputs must be divisible by max_num_modules")
            self.modular_act_dim = num_outputs // max_num_modules
            self.aggregate = False

        self.input_projection_modular = nn.Linear(self.modular_obs_dim, embedding_dim)
        self.input_projection_global = nn.Linear(num_global_obs, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dropout=dropout,
            activation=activation,
            dim_feedforward=embedding_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.action_head = nn.Linear(embedding_dim, self.modular_act_dim)

        # initialize weights
        nn.init.xavier_uniform_(self.input_projection_modular.weight)
        nn.init.xavier_uniform_(self.input_projection_global.weight)
        nn.init.xavier_uniform_(self.action_head.weight)
        nn.init.constant_(self.input_projection_modular.bias, 0.0)
        nn.init.constant_(self.input_projection_global.bias, 0.0)
        nn.init.constant_(self.action_head.bias, 0.0)


    def forward(self, modular_obs: torch.Tensor, global_obs: torch.Tensor, module_masks=None) -> torch.Tensor:
        """Compute actions from modular and global observations.

        Args:
            modular_obs: Tensor of shape ``[batch_size, num_obs]`` containing all modular
                observations concatenated.
            global_obs: Tensor of shape ``[batch_size, num_global_obs]``.

        Returns:
            Tensor of shape ``[batch_size, num_outputs]`` containing outputs for all modules.
        """
        batch_size = modular_obs.shape[0]
        modular_obs = modular_obs.view(batch_size, self.max_num_modules, self.modular_obs_dim)
        feature_modular = self.input_projection_modular(modular_obs) # [batch_size, max_num_modules, embedding_dim]
        feature_global = self.input_projection_global(global_obs).unsqueeze(1) # [batch_size, 1, embedding_dim]
        tokens = torch.cat([feature_global, feature_modular], dim=1)

        if module_masks is not None:
            key_padding_mask = torch.cat(
                [
                    torch.zeros(module_masks.shape[0], 1, dtype=torch.bool, device=self.device),
                    ~module_masks,
                ],
                dim=1,
            )
        else:
            key_padding_mask = None

        encoded = self.transformer(tokens, src_key_padding_mask=key_padding_mask) 
        action_tokens = encoded[:, 1:, :]
        actions = self.action_head(action_tokens) # [batch_size, max_num_modules, modular_act_dim]
        if self.aggregate:
            return actions.mean(dim=1)
        return actions.flatten(start_dim=1)
