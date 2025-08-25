import math
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
        num_layers=1,
    ):
        super().__init__()
        self.modular_obs_dim = modular_obs_dim
        self.max_num_modulars = max_num_modulars
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.device = device
        self.num_layers = num_layers

        self.input_projection_modular = nn.Linear(modular_obs_dim, embedding_dim)
        self.input_projection_global = nn.Linear(global_obs_dim, embedding_dim)

        self.gate = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, num_experts),
        )

        self.posi = nn.Embedding(self.max_num_modulars + 1, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize gate networks with Xavier uniform and zero biases.

        The final linear layer before the softmax is initialized to zeros so that the
        initial gating distribution is uniform over experts.
        """

        linear_layers = [
            self.input_projection_modular,
            self.input_projection_global,
        ]
        for layer in linear_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        for module in self.gate.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        # start with uniform gating probabilities
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, modular_obs, global_obs, module_masks=None):
        # modular_obs: [batch_size, max_num_modules, modular_obs_dim]
        # global_obs: [batch_size, global_obs_dim]
        feature_modular = self.input_projection_modular(modular_obs)  # [batch_size, max_num_modules, embedding_dim]
        feature_global = self.input_projection_global(global_obs)  # [batch_size, embedding_dim]
        feature_integration = torch.cat([
            feature_global.unsqueeze(1),
            feature_modular,
        ], dim=1)   # [batch_size, max_num_modules + 1, embedding_dim]

        # add positional encoding
        posi_indices = torch.arange(
            self.max_num_modulars + 1, device=self.device
        )
        feature_integration += self.posi(posi_indices).unsqueeze(0)  # [1, max_num_modules + 1, embedding_dim]

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

        encoded = self.encoder(
            feature_integration,
            src_key_padding_mask=key_padding_mask,
        )  # [batch_size, max_num_modules + 1, embedding_dim]

        logits = self.gate(encoded[:, 1:, :])
        # logits: [batch_size, max_num_modules, num_experts]
        gate = torch.softmax(logits, dim=-1)

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
        gate_num_layers,
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
                num_layers=gate_num_layers,
                dropout=gate_dropout,
                num_experts=num_experts,
                device=self.device,
            )
        else:
            raise ValueError(
                f"Unknown global encoder type: {gate_type}. Should be 'linear' or 'attention'."
            )

        self._init_parameters()

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

        # compute load balance loss to encourage uniform expert usage
        lb_loss = self._load_balance_loss(gate, module_masks)

        gate_mean = gate.mean(dim=(0, 1))

        return output, lb_loss, gate_mean

    def _init_parameters(self):
        """Initialize experts and gate networks."""

        for expert in self.act_experts:
            for module in expert.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                    nn.init.zeros_(module.bias)

        if self.gate_type == "linear":
            for module in self.gate:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
            # uniform gating at initialization
            nn.init.zeros_(self.gate[-2].weight)
            nn.init.zeros_(self.gate[-2].bias)

    def _load_balance_loss(self, gate, module_masks=None):
        """Compute load-balance loss to encourage uniform expert usage.

        Args:
            gate (torch.Tensor):
                Gating probabilities with shape ``[batch_size, max_num_modules, num_experts]``.
            module_masks (torch.Tensor | None):
                Boolean mask indicating valid modules with shape
                ``[batch_size, max_num_modules]``. ``True`` denotes a valid module.

        Returns:
            torch.Tensor: Scalar load-balance loss.
        """

        if module_masks is not None:
            mask = module_masks.unsqueeze(-1).float()
            gate = gate * mask
            num_tokens = mask.sum()
            gate_mean = gate.sum(dim=(0, 1)) / (num_tokens + 1e-9)
        else:
            gate_mean = gate.mean(dim=(0, 1))

        loss = torch.mean(gate_mean * gate_mean) * (self.num_experts ** 2)
        return loss
