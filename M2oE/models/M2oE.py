import torch
import torch.nn as nn


class M2oEGate(nn.Module):
    def __init__(
        self,
        modular_obs_dim,
        global_obs_dim,
        max_num_modulars,
        embedding_dim,
        num_heads,
        dropout,
        num_experts,
        device,
    ):
        super().__init__()
        self.modular_obs_dim = modular_obs_dim
        self.max_num_modulars = max_num_modulars
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.device = device

        self.input_projection_modular = nn.Linear(modular_obs_dim, embedding_dim)
        self.input_projection_global = nn.Linear(global_obs_dim, embedding_dim)

        self.q_projection = nn.Linear(embedding_dim, embedding_dim)
        self.k_projection = nn.Linear(embedding_dim, embedding_dim)
        self.v_projection = nn.Linear(embedding_dim, embedding_dim)

        self.gate = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, num_experts),
            nn.Softmax(dim=-1)
        )

        self.norm = nn.LayerNorm(embedding_dim)

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, modular_obs, global_obs):
        # modular_obs: [batch_size, max_num_modules, modular_obs_dim]
        # global_obs: [batch_size, global_obs_dim]
        feature_modular = self.input_projection_modular(modular_obs)  # [batch_size, max_num_modules, embedding_dim]
        feature_global = self.input_projection_global(global_obs)  # [batch_size, embedding_dim]

        feature_integration = torch.cat([
            feature_global.unsqueeze(1),
            feature_modular,
        ], dim=1)   # [batch_size, max_num_modules + 1, embedding_dim]

        # Compute attention scores
        q = self.q_projection(feature_integration)
        k = self.k_projection(feature_integration)
        v = self.v_projection(feature_integration)

        # q, k, v: [batch_size, max_num_modules + 1, embedding_dim]
        attn_output, _ = self.multihead_attn(q, k, v)   # [batch_size, max_num_modules + 1, embedding_dim]
        attn_output = self.norm(attn_output + feature_integration)  # Residual connection

        gate = self.gate(attn_output[:, 1:, :])

        return gate


class M2oE(nn.Module):
    def __init__(
        self,
        num_obs,
        num_global_obs,
        max_num_modules,
        num_outputs,
        hidden_dim,
        num_experts,
        activation,
        gate_type,
        gate_embedding_dim,
        gate_num_heads,
        gate_dropout,
        device,
    ):
        super().__init__()

        self.device = device

        self.num_obs = num_obs
        self.num_global_obs = num_global_obs
        self.num_outputs = num_outputs
        self.num_experts = num_experts
        self.activation = activation
        self.max_num_modules = max_num_modules
        self.modular_obs_dim = num_obs // max_num_modules
        if num_outputs == 1:
            self.modular_act_dim = 1
        else:
            assert (
                num_outputs % max_num_modules == 0
            ), "num_outputs must be divisible by max_num_modules"
            self.modular_act_dim = num_outputs // max_num_modules

        # initialize
        self.act_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.modular_obs_dim, hidden_dim),
                self.activation,
                nn.Linear(hidden_dim, hidden_dim),
                self.activation,
                nn.Linear(hidden_dim, self.modular_act_dim)
            ) for _ in range(num_experts)
        ])

        if gate_type == "linear":
            self.gate = nn.Sequential(
                nn.Linear(num_global_obs, hidden_dim),
                self.activation,
                nn.Linear(hidden_dim, hidden_dim),
                self.activation,
                nn.Linear(hidden_dim, hidden_dim)
            )
        elif gate_type == "attention":
            self.gate = M2oEGate(
                modular_obs_dim=self.modular_obs_dim,
                global_obs_dim=num_global_obs,
                max_num_modulars=max_num_modules,
                embedding_dim=gate_embedding_dim,
                num_heads=gate_num_heads,
                dropout=gate_dropout,
                num_experts=num_experts,
                device=self.device,
            )
        else:
            raise ValueError(
                f"Unknown global encoder type: {gate_type}. Should be 'linear' or 'attention'."
            )

        # self.gate = nn.Sequential(
        #     nn.Linear(self.modular_obs_dim + hidden_dim, hidden_dim),
        #     self.activation,
        #     nn.Linear(hidden_dim, num_experts),
        #     nn.Softmax(dim=-1)
        # )

    def forward(self, obs, global_obs):
        # obs: [batch_size, num_obs_padded]
        batch_size = obs.shape[0]   # batch_size = num_envs * num_morphologies
        # obs: [batch_size, num_obs_padded] -> [batch_size, max_num_modules, modular_obs_dim]
        obs = obs.reshape(batch_size, self.max_num_modules, -1)

        expert_outputs = []
        for i in range(self.num_experts):
            expert_output = self.act_experts[i](obs)
            expert_outputs.append(expert_output)
        # expert_outputs: [batch_size, max_num_modules, num_experts, modular_act_dim]
        expert_outputs = torch.stack(expert_outputs, dim=2)

        # global feature extraction
        # [batch_size, num_global]
        # global_obs
        # global_features = self.gate(global_obs).reshape(batch_size, -1)

        # gate compute
        # gate_input: [batch_size, max_num_modules, modular_obs_dim + hidden_dim]
        # gate_input: modular_obs + global_features
        # global_features = global_features.unsqueeze(1).expand(-1, self.max_num_modules, -1)
        # gate_input = torch.cat((obs, global_features), dim=-1)
        # gate: [batch_size, max_num_modules, num_experts]
        # gate = self.gate(gate_input)
        gate = self.gate(obs, global_obs)
        # gate: [batch_size, max_num_modules, num_experts]

        # construct output
        # output: [batch_size, max_num_modules, modular_act_dim]
        output = torch.einsum('bmn, bmnk -> bmk', gate, expert_outputs)
        if self.num_outputs == 1:
            output = output.mean(dim=1)
        else:
            output = output.flatten(start_dim=1)

        return output
