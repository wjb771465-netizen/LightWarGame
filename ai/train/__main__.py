"""训练入口：python -m ai.train --scenario 1v1/vsbaseline"""

from ai.train.args import get_config
from ai.train.sb3_trainer import _set_seeds, train


def main() -> None:
    parser = get_config()
    args = parser.parse_args()
    _set_seeds(args.seed)
    if args.self_play:
        from ai.train.self_play_trainer import train_self_play
        train_self_play(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
