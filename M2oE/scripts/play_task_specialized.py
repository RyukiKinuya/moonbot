"""Play one stream from a task-specialized nine-policy M2oE bank."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
from specialized_play_common import add_stream_args  # noqa: E402
add_stream_args(parser, "Integration_Locomotion_v1", "minimal")
parser.add_argument("--num_steps", type=int, default=1000)
parser.add_argument("--real_time", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

from specialized_play_common import rollout  # noqa: E402

if __name__ == "__main__":
    for _ in rollout(args, app, args.num_steps):
        pass
    app.close()
