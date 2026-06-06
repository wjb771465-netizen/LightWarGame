"""训练入口：python -m ai.train --scenario 1v1/vsbaseline"""

from ai.train.args import get_config
from ai.train.sb3_trainer import Sb3Trainer
from ai.train.self_play_trainer import SelfPlayTrainer


def main() -> None:
    args = get_config().parse_args()
    trainer = SelfPlayTrainer(args) if args.self_play else Sb3Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
