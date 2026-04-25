"""
训练参数配置。

用法：
    from ai.args import get_config
    parser = get_config()
    args = parser.parse_args()

--scenario 决定 env 配置路径，并作为 checkpoint 目录的默认后缀：
    ai/checkpoints/<scenario>/
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
    parser = _get_render_config(parser)
    parser = _get_log_config(parser)
    return parser


# ---------------------------------------------------------------------------
# Prepare
# ---------------------------------------------------------------------------

def _get_prepare_config(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Prepare parameters:
        --scenario <str>
            env 配置名，路径相对于 ai/envs/configs/，如 two_players/vsbaseline。
            同时作为 checkpoint 目录的默认后缀：ai/checkpoints/<scenario>/
        --seed <int>
            random / numpy / torch 的全局种子（default: 42）
    """
    group = parser.add_argument_group("Prepare parameters")
    group.add_argument(
        "--scenario", type=str, required=True,
        help="env 配置名，如 two_players/vsbaseline（同时决定默认 checkpoint 路径）",
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
    return parser


# ---------------------------------------------------------------------------
# Save & Checkpoint
# ---------------------------------------------------------------------------

def _get_save_config(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Save parameters:
        --save-dir <str>
            checkpoint 根目录；默认 ai/checkpoints/<scenario>
        --checkpoint-freq <int>
            每隔多少步保存一次 checkpoint（default: 100_000）
    """
    group = parser.add_argument_group("Save parameters")
    group.add_argument(
        "--save-dir", type=str, default=None,
        help="checkpoint 根目录；不传则自动解析为 ai/checkpoints/<scenario>",
    )
    group.add_argument("--checkpoint-freq", type=int, default=100_000,
                       help="每隔多少步保存一次 checkpoint（default: 100_000）")
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
# Render
# ---------------------------------------------------------------------------

def _get_render_config(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Render parameters:
        --render
            推理模式：加载 <save-dir>/final.zip，运行完整对局并存图。
            不传则走正常训练流程。
        --render-episodes <int>
            render 模式下运行的局数（default: 1）
    """
    group = parser.add_argument_group("Render parameters")
    group.add_argument("--render", action="store_true", default=False,
                       help="推理模式：加载 final.zip，运行完整对局并存图至 <save-dir>/renders/")
    group.add_argument("--render-episodes", type=int, default=1,
                       help="render 模式下运行的局数（default: 1）")
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
    """
    group = parser.add_argument_group("Log parameters")
    group.add_argument("--wandb", action="store_true", default=False,
                       help="启用 W&B 日志上报（default: False）")
    group.add_argument("--wandb-project", type=str, default=None,
                       help="W&B 项目名；默认从 --scenario 外层目录解析")
    group.add_argument("--exp-name", type=str, default=None,
                       help="W&B run 名称；默认从 --scenario 内层名解析")
    return parser


if __name__ == "__main__":
    parser = get_config()
    all_args = parser.parse_args()
    print(all_args)
