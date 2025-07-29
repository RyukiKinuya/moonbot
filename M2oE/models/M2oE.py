import torch
import torch.nn as nn

from M2oE.configs import morphology_configs
from M2oE.utils.utils import djikstra_all_pairs


class GraphAttention(nn.Module):
    """Graph attention encoder for module relations."""

    def __init__(self, adjacency: torch.Tensor, num_heads: int, d_model: int, device: str) -> None:
        super().__init__()
        self.device = device
        self.adjacency = adjacency.to(self.device)
        self.num_nodes = self.adjacency.shape[0]
        self.num_heads = num_heads
        self.d_model = d_model

        # initialize encodings
        self.spatial_encoding_raw = self._init_parameters().to(torch.int)

        # parameters
        self.degree_embedding = nn.Embedding(self.num_degree, self.d_model, padding_idx=0).to(self.device)
        num_spatial_encoding = len(torch.unique(self.spatial_encoding_raw))
        self.spatial_embedding = nn.Embedding(num_spatial_encoding, self.num_heads, padding_idx=0).to(self.device)

    def _init_parameters(self):
        with torch.no_grad():
            self.degree = self.adjacency.sum(dim=1)
            self.degree = self.degree.to(torch.int)
            self.num_degree = int(torch.max(self.degree).item()) + 1

            self.SPD = djikstra_all_pairs(self.adjacency)
            spatial_encoding_raw = torch.zeros(self.num_nodes, self.num_nodes)
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    if i == j:
                        continue
                    spatial_encoding_raw[i, j] = self.SPD[i][j][1]

            max_weight_num = 0
            for start in range(self.num_nodes):
                for end in range(start):
                    max_weight_num = max(max_weight_num, self.SPD[start][end][1])

            self.feature_weight = nn.Parameter(torch.zeros(self.num_nodes, self.num_nodes, self.num_heads, max_weight_num)).to(self.device)

        return spatial_encoding_raw

    def forward(self):
        # degree encoding
        degree_encoding = self.degree_embedding(self.degree)
        degree_encoding.to(self.device)

        # spatial encoding
        spatial_encoding = self.spatial_embedding(self.spatial_encoding_raw)
        spatial_encoding = spatial_encoding.permute(2, 0, 1)
        spatial_encoding.to(self.device)

        # feature encoding
        feature_encoding = torch.zeros(self.num_nodes, self.num_nodes, self.num_heads).to(self.device)
        for start in range(self.num_nodes):
            for end in range(self.num_nodes):
                if start == end:
                    continue
                weight_num = self.SPD[start][end][1]
                weight = self.feature_weight[start, end, :, :weight_num]
                feature_encoding[start, end] = torch.sum(weight, dim=-1)

        feature_encoding = feature_encoding.permute(2, 0, 1)
        return degree_encoding, spatial_encoding, feature_encoding


class M2oEGate(nn.Module):
    def __init__(
        self,
        modular_obs_dim,
        global_obs_dim,
        max_num_modulars,
        embedding_dim,
        num_heads,
        d_model,
        dim_feedforward,
        dropout,
        num_experts,
        device="cpu",
    ):
        super().__init__()
        self.modular_obs_dim = modular_obs_dim
        self.max_num_modulars = max_num_modulars
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.d_model = d_model
        self.num_experts = num_experts
        self.device = device

        self.input_projection_modular = nn.Linear(modular_obs_dim, embedding_dim)
        self.input_projection_global = nn.Linear(global_obs_dim, embedding_dim)

        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(self.layer, num_layers=1)

        base_adj = torch.tensor(
            morphology_configs.adjacency_mat_dict[morphology_configs.morphology_list[-1]],
            dtype=torch.int,
        )
        full_adj = torch.zeros(max_num_modulars + 1, max_num_modulars + 1, dtype=torch.int)
        full_adj[1:, 1:] = base_adj
        full_adj[0, 1:] = 1
        full_adj[1:, 0] = 1
        self.graph_attention = GraphAttention(
            adjacency=full_adj,
            num_heads=num_heads,
            d_model=d_model,
            device=self.device,
        ).to(self.device)

        self.gate_layer = nn.Linear(d_model, num_experts)

    def forward(self, modular_obs, global_obs):
        degree_encoding, spatial_encoding, feature_encoding = self.graph_attention()
        attn_encoding = spatial_encoding + feature_encoding

        batch_size = modular_obs.shape[0]

        modular_obs = self.input_projection_modular(modular_obs)
        global_obs = self.input_projection_global(global_obs).unsqueeze(1)

        x = torch.cat([global_obs, modular_obs], dim=1)
        x = x + degree_encoding.unsqueeze(0)
        attn_mask = (
            attn_encoding.unsqueeze(0)
            .repeat(batch_size, 1, 1, 1)
            .reshape(-1, self.graph_attention.num_nodes, self.graph_attention.num_nodes)
        )

        x = self.transformer(x, mask=attn_mask)

        gate_logits = self.gate_layer(x[:, 1:, :])
        gate = torch.softmax(gate_logits, dim=-1)

        return gate


class M2oE(nn.Module):
    def __init__(
        self,
        num_obs,
        num_global_obs,
        hidden_dim,
        max_num_modules,
        num_actions,
        num_experts,
        activation,
        global_encoder_type="linear",
        device="cpu",
    ):
        super().__init__()

        self.device = device

        self.num_obs = num_obs
        self.num_global_obs = num_global_obs
        self.num_actions = num_actions
        self.num_experts = num_experts
        self.activation = activation
        self.max_num_modules = max_num_modules
        self.modular_obs_dim = num_obs // max_num_modules
        self.modular_act_dim = num_actions // max_num_modules
        self.graph_atten_gate = M2oEGate(
            modular_obs_dim=self.modular_obs_dim,
            global_obs_dim=num_global_obs,
            max_num_modulars=max_num_modules,
            embedding_dim=hidden_dim,
            num_heads=4,  # Example value, can be adjusted
            d_model=hidden_dim,
            dim_feedforward=hidden_dim * 4,  # Example value, can be adjusted
            dropout=0.1,  # Example value, can be adjusted
            num_experts=num_experts,
            device=self.device,
        )

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

        if global_encoder_type == "linear":
            self.global_feature_extractor = nn.Sequential(
                nn.Linear(num_global_obs, hidden_dim),
                self.activation,
                nn.Linear(hidden_dim, hidden_dim),
                self.activation,
                nn.Linear(hidden_dim, hidden_dim)
            )
        else:
            raise ValueError(
                f"Unknown global encoder type: {global_encoder_type}. Should be 'linear' or 'attention'."
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
        # global_obs
        # global_features = self.global_feature_extractor(global_obs).reshape(batch_size, -1)

        # gate compute
        # gate_input: [batch_size, max_num_modules, modular_obs_dim + hidden_dim]
        # gate_input: modular_obs + global_features
        # global_features = global_features.unsqueeze(1).expand(-1, self.max_num_modules, -1)
        # gate_input = torch.cat((obs, global_features), dim=-1)
        # gate: [batch_size, max_num_modules, num_experts]
        # gate = self.gate(gate_input)
        gate = self.graph_atten_gate(modular_obs=obs, global_obs=global_obs)

        # contruct action
        # action: [batch_size, max_num_modules, num_actions] -> [batch_size, num_actions]
        action = torch.einsum('bmn, bmnk -> bmk', gate, expert_outputs)
        action = action.flatten(start_dim=1)
        # action: [batch_size, num_actions]

        return action
