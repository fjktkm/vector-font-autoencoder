import sys

from absl import app
from tensorboard import default, main_lib, program
from tensorboard.plugins import base_plugin


def main() -> None:
    main_lib.global_init()
    tensorboard = program.TensorBoard(plugins=default.get_plugins())
    argv = ["tensorboard", "--logdir", "."]
    try:
        app.run(tensorboard.main, flags_parser=tensorboard.configure, argv=argv)
    except base_plugin.FlagsError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
