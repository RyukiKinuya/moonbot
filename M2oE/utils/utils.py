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
    # obs: [num_envs * num_morphologies, num_obs]
    return obs, global_obs
