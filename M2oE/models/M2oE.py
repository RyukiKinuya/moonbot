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

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embedding_dim)

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, modular_obs, global_obs, module_masks=None):
        # modular_obs: [batch_size, max_num_modules, modular_obs_dim]
        # global_obs: [batch_size, global_obs_dim]
        feature_modular = self.input_projection_modular(modular_obs)  # [batch_size, max_num_modules, embedding_dim]
        feature_global = self.input_projection_global(global_obs)  # [batch_size, embedding_dim]

        feature_integration = torch.cat([
            feature_global.unsqueeze(1),
            feature_modular,
        ], dim=1)   # [batch_size, max_num_modules + 1, embedding_dim]

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

        q = self.q_projection(feature_integration)  # [batch_size, max_num_modules + 1, embedding_dim]
        k = self.k_projection(feature_integration)
        v = self.v_projection(feature_integration)

        attn_output, _ = self.multihead_attn(
            query=q,
            key=k,
            value=v,
            key_padding_mask=key_padding_mask,
        )  # [batch_size, max_num_modules + 1, embedding_dim]
        attn_output = self.dropout(self.norm(attn_output))  # Residual connection

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
        top_k=2,
    ):
        super().__init__()

        self.device = device

        self.top_k = top_k
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
                nn.Linear(self.modular_obs_dim + self.num_global_obs, hidden_dim),
                self.activation,
                nn.Linear(hidden_dim, hidden_dim),
                self.activation,
                nn.Linear(hidden_dim, self.modular_act_dim)
            ) for _ in range(num_experts)
        ])

        self.gate_type = gate_type
        if gate_type == "linear":
            self.gate = nn.Sequential(
                nn.Linear(
                    max_num_modules + self.modular_obs_dim + num_global_obs,
                    hidden_dim,
                ),
                self.activation,
                nn.Linear(hidden_dim, hidden_dim),
                self.activation,
                nn.Linear(hidden_dim, num_experts),
                nn.Softmax(dim=-1),
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


    def forward(self, obs, global_obs, module_masks=None):
        # obs: [batch_size, num_obs_padded]
        batch_size = obs.shape[0]   # batch_size = num_envs * num_morphologies
        # obs: [batch_size, num_obs_padded] -> [batch_size, max_num_modules, modular_obs_dim]
        obs = obs.reshape(batch_size, self.max_num_modules, -1)
        if module_masks is not None:
            module_masks = module_masks.reshape(batch_size, self.max_num_modules)
        # global_obs: [batch_size, num_global_obs] -> [batch_size, self.max_num_modules, num_global_obs]
        expert_global_obs = global_obs.unsqueeze(1).expand(-1, self.max_num_modules, -1)
        expert_input = torch.cat((obs, expert_global_obs), dim=-1)

        expert_outputs = []
        for i in range(self.num_experts):
            expert_output = self.act_experts[i](expert_input)
            expert_outputs.append(expert_output)
        # expert_outputs: [batch_size, max_num_modules, num_experts, modular_act_dim]
        expert_outputs = torch.stack(expert_outputs, dim=2)

        if self.gate_type == "linear":
            module_onehot = torch.eye(self.max_num_modules, device=self.device)
            module_onehot = module_onehot.unsqueeze(0).expand(batch_size, -1, -1)
            gate_input = torch.cat((module_onehot, obs, expert_global_obs), dim=-1)
            gate = self.gate(gate_input)
        elif self.gate_type == "attention":
            gate = self.gate(obs, global_obs, module_masks)
        # gate: [batch_size, max_num_modules, num_experts]

        # construct output
        # output: [batch_size, max_num_modules, modular_act_dim]
        output = torch.einsum('bmn, bmnk -> bmk', gate, expert_outputs)
        if self.num_outputs == 1:
            output = output.mean(dim=1)
        else:
            output = output.flatten(start_dim=1)

        return output
