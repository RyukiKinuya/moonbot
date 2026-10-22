"""Collect gate activations for one task-specialized stream."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
from specialized_play_common import add_stream_args  # noqa: E402
add_stream_args(parser, "Integration_Locomotion_v1", "minimal")
parser.add_argument("--num_steps", type=int, default=1000)
parser.add_argument("--heatmap_path", default="task_specialized_gate_heatmap.png")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import torch  # noqa: E402
from specialized_play_common import gate_values, rollout, save_gate_heatmap  # noqa: E402

if __name__ == "__main__":
    values = []
    for value in rollout(args, app, args.num_steps,
                         callback=lambda _, p, o, g, m, __: gate_values(p, o, g, m)):
        values.append(value.detach().cpu())
    if values:
        save_gate_heatmap(torch.cat(values), args.heatmap_path,
                          f"Task-specialized gate: {args.task_kind}/{args.morphology}")
    app.close()
