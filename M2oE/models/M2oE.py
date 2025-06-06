import torch
import torch.nn as nn


class M2oE(nn.Module):
    def __init__(self, num_obs, num_global_obs, hidden_dim, max_num_modules,
                 num_actions, num_experts, activation):
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
                self.activation(),
                nn.Linear(hidden_dim, self.modular_act_dim)
            ) for _ in range(num_experts)
        ])

        self.global_feature_extractor = nn.Sequential(
            nn.Linear(num_global_obs + num_obs, hidden_dim),
            self.activation(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.gate = nn.Sequential(
            nn.Linear(self.modular_obs_dim + hidden_dim, hidden_dim),
            self.activation(),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1)
        )


    def forward(self, obs, global_obs):
        # obs: [batch_size, max_num_modules, modular_obs_dim]
        batch_size = obs.shape[0]
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
        # [batch_size, hidden_dim]
        global_features = self.global_feature_extractor(torch.cat([obs, global_obs], dim=-1))

        # gate compute
        # gate_input: [batch_size, max_num_modules, modular_obs_dim + hidden_dim]
        # gate_input: modular_obs + global_features
        gate_input = torch.cat([obs, global_features.unsqueeze(1).expand(-1, self.max_num_modules, -1)], dim=-1)
        # gate: [batch_size, max_num_modules, num_experts]
        gate = self.gate(gate_input)

        # contruct action
        # action: [batch_size, max_num_modules, num_actions] -> [batch_size, num_actions]
        action = torch.einsum('bmn, bmnk -> bmk', gate, expert_outputs)
        action = action.flatten(start_dim=1)

        return action
