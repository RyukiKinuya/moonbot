import torch

from M2oE.configs import morphology_configs


def process_observations(obs_dict, num_obs, num_global_obs, policy, num_envs):
    global_name_list = [f"global_obs_{i}" for i in morphology_configs.morphology_list]
    try:
        global_obs = [obs_dict[name] for name in global_name_list]
    except KeyError as e:
        raise KeyError(f"Observation dictionary does not contain expected keys: {global_name_list}.") from e

    global_obs = torch.cat(global_obs, dim=1).reshape(-1, num_global_obs)

    for key in global_name_list:
        obs_dict.pop(key, None)  # remove global observations from obs_dict

    padded_obs = []
    pad_vec = policy.padding
    for morph in morphology_configs.morphology_list:
        key = f"obs_{morph.replace('moonbot_', '')}"
        obs = obs_dict[key]
        diff = num_obs - obs.shape[1]
        if policy.padding_method == "concat":
            if diff > 0:
                pad = pad_vec[obs.shape[1] : num_obs].unsqueeze(0).expand(obs.shape[0], -1)
                # obs: [num_envs, num_obs]
                obs = torch.cat([obs, pad], dim=1)
        elif policy.padding_method == "add":
            pad = pad_vec.expand(obs.shape[0], -1)
            obs = obs + pad

        # padded_obs: [num_envs, 1, num_obs]
        padded_obs.append(obs.unsqueeze(1))

    obs = torch.cat(padded_obs, dim=1).reshape(num_envs, num_obs)
    return obs, global_obs


def djikstra_all_pairs(adjacency):
    num_nodes = adjacency.shape[0]
    all_shortest_paths = {}

    for start_node in range(num_nodes):
        shortest_paths = {}
        predecessors = {}
        visited = set()
        visited.add(start_node)
        shortest_paths[start_node] = 0
        predecessors[start_node] = None

        while len(visited) < num_nodes:
            min_node = None
            min_dist = float('inf')
            for node in visited:
                dist = shortest_paths[node]
                for neighbor in range(num_nodes):
                    if neighbor not in visited and adjacency[node, neighbor] == 1:
                        new_dist = dist + 1
                        if new_dist < min_dist:
                            min_node = neighbor
                            min_dist = new_dist
                            predecessors[min_node] = node

            shortest_paths[min_node] = min_dist
            visited.add(min_node)

        paths = {}
        for end_node in range(num_nodes):
            if end_node != start_node:
                path = []
                current_node = end_node
                while current_node is not None:
                    path.insert(0, current_node)
                    current_node = predecessors[current_node]
                path_length = shortest_paths[end_node]
                paths[end_node] = (path, path_length)

        all_shortest_paths[start_node] = paths

    return all_shortest_paths
