"""
训练参数配置。

用法：
    from ai.train.args import get_config
    parser = get_config()
    args = parser.parse_args()

--scenario 决定 env 配置路径，并作为结果目录的默认后缀：
    ai/train/results/<scenario>/
"""
from __future__ import annotations

import argparse


def get_config() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train MaskablePPO on LightWarGame",
    )
    parser = _get_prepare_config(parser)
    parser = _get_ppo_config(parser)
    parser = _get_save_config(parser)
    parser = _get_eval_config(parser)
    parser = _get_log_config(parser)
    parser = _get_self_play_config(parser)
    parser = _get_region_self_play_config(parser)
    return parser


# ---------------------------------------------------------------------------
# Prepare
# ---------------------------------------------------------------------------

def _get_prepare_config(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Prepare parameters:
        --scenario <str>
            env 配置名，路径相对于 ai/envs/configs/，如 two_players/vsbaseline。
            同时作为结果目录的默认后缀：ai/train/results/<scenario>/
        --seed <int>
            random / numpy / torch 的全局种子（default: 42）
    """
    group = parser.add_argument_group("Prepare parameters")
    group.add_argument(
        "--scenario", type=str, required=True,
        help="env 配置名，如 two_players/vsbaseline（同时决定默认结果路径）",
    )
    group.add_argument(
        "--seed", type=int, default=42,
        help="全局随机种子（default: 42）",
    )
    return parser


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------

def _get_ppo_config(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    PPO parameters:
        --total-timesteps <int>     总训练步数（default: 1_000_000）
        --n-steps <int>             每次 rollout 采集步数（default: 2048）
        --batch-size <int>          minibatch 大小（default: 64）
        --n-epochs <int>            每批数据重复训练轮数（default: 10）
        --lr <float>                学习率（default: 3e-4）
        --gamma <float>             折扣因子（default: 0.99）
        --gae-lambda <float>        GAE lambda（default: 0.95）
        --clip-range <float>        PPO clip 系数（default: 0.2）
        --net-arch <int ...>        MLP 隐层大小序列（default: 256 256）
    """
    group = parser.add_argument_group("PPO parameters")
    group.add_argument("--total-timesteps", type=int, default=1_000_000,
                       help="总训练步数（default: 1_000_000）")
    group.add_argument("--n-steps", type=int, default=2048,
                       help="每次 rollout 采集步数（default: 2048）")
    group.add_argument("--batch-size", type=int, default=64,
                       help="minibatch 大小（default: 64）")
    group.add_argument("--n-epochs", type=int, default=10,
                       help="每批数据重复训练轮数（default: 10）")
    group.add_argument("--lr", type=float, default=3e-4,
                       help="学习率（default: 3e-4）")
    group.add_argument("--gamma", type=float, default=0.99,
                       help="折扣因子（default: 0.99）")
    group.add_argument("--gae-lambda", type=float, default=0.95,
                       help="GAE lambda（default: 0.95）")
    group.add_argument("--clip-range", type=float, default=0.2,
                       help="PPO clip 系数（default: 0.2）")
    group.add_argument("--net-arch", type=int, nargs="+", default=[256, 256],
                       help="MLP 隐层大小序列（default: 256 256）")
    group.add_argument("--n-envs", type=int, default=4,
                       help="训练 rollout 并行环境数（default: 4）")
    group.add_argument("--n-opponents", type=int, default=None,
                       help="每 chunk 对手种类数，--n-envs 须为其整数倍，多个 env 可共享同一对手（default: 与 --n-envs 相同）")
    return parser


# ---------------------------------------------------------------------------
# Save & Checkpoint
# ---------------------------------------------------------------------------

def _get_save_config(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Save parameters:
        --save-dir <str>
            结果根目录；默认 ai/train/results/<scenario>
        --checkpoint-freq <int>
            每隔多少步保存一次 checkpoint（default: 100_000）
    """
    group = parser.add_argument_group("Save parameters")
    group.add_argument(
        "--save-dir", type=str, default=None,
        help="结果根目录；不传则自动解析为 ai/train/results/<scenario>",
    )
    group.add_argument("--checkpoint-freq", type=int, default=100_000,
                       help="每隔多少步保存一次 checkpoint（default: 100_000）")
    group.add_argument("--resume-from", type=str, default=None,
                       help="从 checkpoint(.zip) 恢复训练，num_timesteps 延续")
    return parser


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def _get_eval_config(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Eval parameters:
        --use-eval                  开启训练过程中的周期评估（default: False）
        --eval-freq <int>           每隔多少步做一次评估（default: 50_000）
        --eval-episodes <int>       每次评估运行的局数（default: 100）
    """
    group = parser.add_argument_group("Eval parameters")
    group.add_argument("--use-eval", action="store_true", default=False,
                       help="开启训练过程中的周期评估（default: False）")
    group.add_argument("--eval-freq", type=int, default=50_000,
                       help="每隔多少步做一次评估（default: 50_000）")
    group.add_argument("--eval-episodes", type=int, default=100,
                       help="每次评估运行的局数（default: 100）")
    return parser


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _get_log_config(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Log parameters:
        --wandb             启用 W&B 日志上报（default: False）
        --wandb-project     W&B 项目名；不传则从 --scenario 外层目录解析
                            （如 two_players/vsbaseline → "two_players"）
        --exp-name          W&B run 名称；不传则从 --scenario 内层名解析
                            （如 two_players/vsbaseline → "vsbaseline"）
        --win-rate-window   胜率滑动窗口局数（default: 100）
    """
    group = parser.add_argument_group("Log parameters")
    group.add_argument("--wandb", action="store_true", default=False,
                       help="启用 W&B 日志上报（default: False）")
    group.add_argument("--wandb-project", type=str, default=None,
                       help="W&B 项目名；默认从 --scenario 外层目录解析")
    group.add_argument("--exp-name", type=str, default=None,
                       help="W&B run 名称；默认从 --scenario 内层名解析")
    group.add_argument("--win-rate-window", type=int, default=100,
                       help="胜率滑动窗口局数（default: 100）")
    return parser


# ---------------------------------------------------------------------------
# Self-Play
# ---------------------------------------------------------------------------

def _get_self_play_config(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Self-Play parameters:
        --self-play                    启用自博弈训练（default: False）
        --self-play-pool-size <int>    策略池最大容量（default: 20）
        --self-play-initial-opponent <str>
                                       冷启动对手类型：random | rule（default: random）
        --pool-sampling-strategy <str>
                                       对手采样策略（default: latest）
                                       latest=最新, uniform=均匀(FSP),
                                       progress=进度优先, elo=ELO优先
        --sampling-lam <float>         Logistic-Softmax 温度系数 λ（default: 1.0）
        --sampling-scale <float>       Logistic-Softmax 缩放因子 s（default: 100.0）
        --progress-D <float>           progress 策略的 logistic 尺度 D，
                                       None=自动取 (max-min)/4（default: None）
    """
    group = parser.add_argument_group("Self-Play parameters")
    group.add_argument("--self-play", action="store_true", default=False,
                       help="启用自博弈训练（default: False）")
    group.add_argument("--self-play-pool-size", type=int, default=20,
                       help="策略池最大容量（default: 20）")
    group.add_argument("--self-play-initial-opponent", type=str, default="random",
                       choices=["random", "rule"],
                       help="冷启动对手类型：random | rule（default: random）")
    group.add_argument("--pool-sampling-strategy", type=str, default="latest",
                       choices=["latest", "uniform", "progress", "elo"],
                       help="对手采样策略（default: latest）")
    group.add_argument("--sampling-lam", type=float, default=1.0,
                       help="Logistic-Softmax 温度系数 λ（default: 1.0）")
    group.add_argument("--sampling-scale", type=float, default=100.0,
                       help="Logistic-Softmax 缩放因子 s（default: 100.0）")
    group.add_argument("--progress-D", type=float, default=None,
                       help="progress 策略的 logistic 尺度 D，不传则自动计算")
    return parser


# ---------------------------------------------------------------------------
# Region Self-Play
# ---------------------------------------------------------------------------

def _get_region_self_play_config(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Region Self-Play parameters:
        --region-self-play                     启用地区自博弈训练（default: False）
        --region-self-play-regions <str>       逗号分隔的地区 ID，如 "4,20"；默认全 31 省
        --parallel-regions <int>               并行训练的地区数（default: 1=串行）。>1 时启用 ThreadPoolExecutor
        --n-training-threads <int>              PyTorch 内部线程数，限制每个地区模型的 CPU 并行度（default: 1）

    地区池大小复用 --self-play-pool-size（每个地区独立维护一个 OpponentPool）。
    """
    group = parser.add_argument_group("Region Self-Play parameters")
    group.add_argument("--region-self-play", action="store_true", default=False,
                       help="启用地区自博弈训练（default: False）")
    group.add_argument("--region-self-play-regions", type=str, default=None,
                       help="逗号分隔的地区 ID，如 '4,20'；默认全 31 省")
    group.add_argument("--parallel-regions", type=int, default=1,
                       help="并行训练的地区数（default: 1=串行）。>1 时启用 ThreadPoolExecutor")
    group.add_argument("--n-training-threads", type=int, default=1,
                       help="PyTorch 内部线程数，限制每个地区模型的 CPU 并行度（default: 1）")
    return parser


if __name__ == "__main__":
    parser = get_config()
    all_args = parser.parse_args()
    print(all_args)
