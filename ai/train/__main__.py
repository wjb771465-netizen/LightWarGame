"""训练入口：python -m ai.train --scenario duel/vsbaseline"""

from ai.train.args import get_config
from ai.train.sb3_trainer import Sb3Trainer
from ai.train.self_play_trainer import SelfPlayTrainer


def main() -> None:
    args = get_config().parse_args()
    if args.region_self_play:
        from ai.train.region_self_play_trainer import RegionSelfPlayTrainer
        trainer = RegionSelfPlayTrainer(args)
    elif args.self_play:
        trainer = SelfPlayTrainer(args)
    else:
        trainer = Sb3Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
