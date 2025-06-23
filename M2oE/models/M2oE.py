import torch
import torch.nn as nn


class M2oE(nn.Module):
    def __init__(self, num_obs, num_global_obs, hidden_dim, max_num_modules,
                 num_actions, num_experts, activation, global_encoder_type="linear"):
        super().__init__()

        self.num_obs = num_obs
        self.num_global_obs = num_global_obs
        self.num_actions = num_actions
        self.num_experts = num_experts
        self.activation = activation
        self.max_num_modules = max_num_modules
        self.modular_obs_dim = num_obs // max_num_modules
        self.modular_act_dim = num_actions // max_num_modules

        # initialize
        self.act_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.modular_obs_dim, hidden_dim),
                self.activation,
                nn.Linear(hidden_dim, self.modular_act_dim)
            ) for _ in range(num_experts)
        ])

        if global_encoder_type == "linear":
            self.global_feature_extractor = nn.Sequential(
                nn.Linear(num_global_obs, hidden_dim),
                self.activation,
                nn.Linear(hidden_dim, hidden_dim)
            )
        elif global_encoder_type == "attention":
            self.global_feature_extractor = nn.Sequential(
                nn.TransformerEncoderLayer(
                    d_model=num_global_obs + num_obs,
                    nhead=8,
                    dim_feedforward=hidden_dim,
                    activation=self.activation.__name__
                ),
                self.activation,
                nn.Linear(hidden_dim, hidden_dim)
            )
        else:
            raise ValueError(f"Unknown global encoder type: {global_encoder_type}. Should be 'linear' or 'attention'."
            )

        self.gate = nn.Sequential(
            nn.Linear(self.modular_obs_dim + hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, obs, global_obs):
        # obs: [batch_size, num_obs_padded]
        batch_size = obs.shape[0]   # batch_size = num_envs * num_morphologies
        # obs: [batch_size, num_obs_padded] -> [batch_size, max_num_modules, modular_obs_dim]
        obs = obs.reshape(batch_size, self.max_num_modules, -1)

        expert_outputs = []
        for i in range(self.num_experts):
            # export_output: [batch_size, max_num_modules, num_actions]
            expert_output = self.act_experts[i](obs)
            expert_outputs.append(expert_output.squeeze(2))
        # expert_outputs: list len:num_experts
        # expert_outputs: [batch_size, max_num_modules, num_experts, num_actions]
        expert_outputs = torch.stack(expert_outputs, dim=2)

        # global feature extraction
        # [batch_size, num_global]
        # global_obs e(f.num_global_obs)
        global_features = self.global_feature_extractor(global_obs).reshape(batch_size, -1)

        # gate compute
        # gate_input: [batch_size, max_num_modules, modular_obs_dim + hidden_dim]
        # gate_input: modular_obs + global_features
        global_features = global_features.unsqueeze(1).expand(-1, self.max_num_modules, -1)
        gate_input = torch.cat((obs, global_features), dim=-1)
        # gate: [batch_size, max_num_modules, num_experts]
        gate = self.gate(gate_input)

        # contruct action
        # action: [batch_size, max_num_modules, num_actions] -> [batch_size, num_actions]
        action = torch.einsum('bmn, bmnk -> bmk', gate, expert_outputs)
        action = action.flatten(start_dim=1)
        # action: [batch_size, num_actions]

        return action
