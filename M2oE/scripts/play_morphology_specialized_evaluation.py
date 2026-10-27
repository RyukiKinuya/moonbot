"""Run the complete Morphology-specialized M2oE evaluation suite."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
from play_common import add_evaluation_args  # noqa: E402
add_evaluation_args(parser, "Integration_Locomotion_v1", allocation="morphology")
parser.add_argument("--num_steps", type=int, default=1000)
parser.add_argument("--heatmap_path", default=None)
parser.add_argument("--pca_path", default=None)
parser.add_argument("--real_time", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

from play_common import run_condition_evaluations  # noqa: E402

if __name__ == "__main__":
    if args.heatmap_path is None:
        args.heatmap_path = f"morphology_specialized_{args.morphology}_evaluation_gate_heatmap.png"
    if args.pca_path is None:
        args.pca_path = f"morphology_specialized_{args.morphology}_evaluation_key_query_pca.png"
    run_condition_evaluations(args, app, args.heatmap_path, args.pca_path)
    app.close()
