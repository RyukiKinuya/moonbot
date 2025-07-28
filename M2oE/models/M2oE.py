import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear

class GraphAttention(nn.Module):
    def __init__(self, num_nodes, parent_map):
        super(GraphAttention, self).__init__()
        self.parent_map = parent_map.to(self.device)
        self.offset = offset.to(self.device)
        self.num_nodes = args.joint_num

        # init 
        self.spatial_encoding_raw = self.init_para().to(torch.int).to(self.device)

        # parameters
        self.degree_embedding = nn.Embedding(self.num_degree, self.args.d_model, padding_idx=0)
        self.degree_embedding = self.degree_embedding.to(self.device)
        num_spatial_encoding = len(set(self.spatial_encoding_raw.reshape(-1).tolist()))
        self.spatial_embedding = nn.Embedding(num_spatial_encoding,  self.args.n_head, padding_idx=0)
        self.spatial_embedding = self.spatial_embedding.to(self.device)

    def init_para(self):
        with torch.no_grad():
            self.degree, self.adjacency = self.compute_degree_and_adjancency()
            self.degree = self.degree.to(torch.int).to(self.device)
            self.num_degree = int(torch.max(self.degree).item()) + 1

            self.adjacency = self.adjacency.to(self.device)
            self.SPD = djikstra_all_pairs(self.adjacency)
            spatial_encoding_raw = torch.zeros( self.parent_map.shape[0],  self.parent_map.shape[0])
            for i in range( self.parent_map.shape[0]):
                for j in range( self.parent_map.shape[0]):
                    if i == j:
                        continue
                    spatial_encoding_raw[i, j] = self.SPD[i][j][1]

            max_weight_num = 0
            for start in range(self.num_nodes):
                for end in range(start):
                    max_weight_num = max(max_weight_num, self.SPD[start][end][1])

            self.offset_weight = nn.Parameter(torch.zeros(self.num_nodes, self.joint_num, self.args.n_head, max_weight_num * 3)).to(self.device)
            self.offset_bias = nn.Parameter(torch.zeros(self.num_nodes, self.joint_num, self.args.n_head, 1)).to(self.device)

        return spatial_encoding_raw

    def forward(self):
        # degree encoding
        degree_encoding = self.degree_embedding(self.degree)
        degree_encoding.to(self.device)

        # spatial encoding
        spatial_encoding = self.spatial_embedding(self.spatial_encoding_raw)
        spatial_encoding = spatial_encoding.permute(2, 0, 1)
        spatial_encoding.to(self.device)

        # offset encoding
        offset_encoding = torch.zeros(self.num_nodes, self.joint_num, self.args.n_head).to(self.device)
        for start in range(self.num_nodes):
            for end in range(self.num_nodes):
                if start == end:
                    continue
                weight_num = self.SPD[start][end][1]
                path = self.SPD[start][end][0]
                
                weight = self.offset_weight[start, end, :, :weight_num * 3]
                bias = self.offset_bias[start, end, :, 0]
                path_feature = torch.cat([self.offset[path[i], path[i+1]] for i in range(len(path) - 1)], dim=0)
                
                offset_encoding[start, end] = torch.matmul(weight, path_feature) + bias / weight_num

        offset_encoding = offset_encoding.permute(2, 0, 1)
        return degree_encoding, spatial_encoding, offset_encoding

        
    def compute_degree_and_adjancency(self):
        degree = torch.zeros(self.args.joint_num)
        adjacency = torch.zeros(self.args.joint_num, self.args.joint_num)

        for i in range(self.args.joint_num):
            if self.parent_map[i] == -1:
                continue
            degree[i] += 1
            degree[self.parent_map[i]] += 1
            adjacency[i, self.parent_map[i]] = 1
            adjacency[self.parent_map[i], i] = 1

        return degree, adjacency
    

# class M2oEGate(nn.Module):
    # def __init__(self, num_obs, max_num_modulars, embedding_dim, num_heads):
        # super().__init__()
        # self.embedding = 




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
