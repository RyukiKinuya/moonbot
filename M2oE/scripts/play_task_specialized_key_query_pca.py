"""Visualize attention-gate query features for one task-specialized stream."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
from specialized_play_common import add_stream_args  # noqa: E402
add_stream_args(parser, "Integration_Locomotion_v1", "minimal")
parser.add_argument("--num_steps", type=int, default=1000)
parser.add_argument("--save_path", default="task_specialized_key_query_pca.png")
parser.add_argument("--max_query_samples", type=int, default=5000)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

from specialized_play_common import extract_queries, rollout, save_query_pca  # noqa: E402

if __name__ == "__main__":
    samples = []
    for value in rollout(args, app, args.num_steps,
                         callback=lambda _, p, o, g, m, __: extract_queries(p, o, g, m)):
        samples.append(value.reshape(-1, value.shape[-1]).detach().cpu())
        if sum(x.shape[0] for x in samples) >= args.max_query_samples:
            break
    if samples:
        save_query_pca([torch.cat(samples)[:args.max_query_samples]], args.save_path,
                       f"Task-specialized query PCA: {args.task_kind}/{args.morphology}")
    app.close()
