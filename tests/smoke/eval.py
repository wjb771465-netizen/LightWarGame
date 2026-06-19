"""Eval 冒烟测试：验证 evaluate() 跑通、结果不崩。

用法:
    conda run -n chinese_war_game python -m tests.smoke.eval \
      --scenario duel/vsbaseline

    conda run -n chinese_war_game python -m tests.smoke.eval \
      --scenario duel/vsbaseline_no_adj --use-gnn

输出: ai/train/results/<scenario>/eval_smoke_<timestamp>.log
"""

from __future__ import annotations

import io
import os
import sys
from datetime import datetime

import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from ai.algos.policy import SB3Policy
from ai.envs.env import LwgEnv
from ai.envs.utils import parse_config
from ai.train.args import get_config
from ai.train.eval import evaluate, aggregate_win_rate, aggregate_avg_turns

# ── 写死冒烟超参 ───────────────────────────────────────────────────────────
SMOKE_DEFAULTS = dict(
    seed=42,
    n_envs=4,
    checkpoint_freq=16384,   # 用来凑一个假 ckpt step
    eval_episodes=8,
    eval_opponent_freq=1,
    eval_opponent="random,rule,fsm",
    use_gnn=False,
    wandb=False,
)

_LOG_STREAM: io.StringIO | None = None
_LOG_PATH: str | None = None
_VERDICT: list[str] = []


def _start_log(filepath: str) -> None:
    global _LOG_STREAM, _LOG_PATH
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    _LOG_STREAM = io.StringIO()
    _LOG_PATH = filepath


def _log(msg: str = "") -> None:
    print(msg, flush=True)
    if _LOG_STREAM is not None:
        _LOG_STREAM.write(msg + "\n")


def _stop_log() -> None:
    global _LOG_STREAM, _LOG_PATH
    if _LOG_STREAM is not None and _LOG_PATH is not None:
        with open(_LOG_PATH, "a") as f:
            f.write(_LOG_STREAM.getvalue())
    _LOG_STREAM = None


def _fail(reason: str) -> None:
    _VERDICT.append(reason)
    _log(f"  FAIL: {reason}")


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = get_config()
    args = parser.parse_args()

    for k, v in SMOKE_DEFAULTS.items():
        setattr(args, k, v)

    _start_log(_log_path(args))
    _log(f"# Eval smoke test — {datetime.now().isoformat()}")
    _log(f"# Scenario: {args.scenario}  use_gnn: {args.use_gnn}")
    _log(f"# n_envs: {args.n_envs}  episodes: {args.eval_episodes}\n")

    cfg = parse_config(args.scenario)
    opponent_id = next(p for p in range(1, cfg.game.num_players + 1) if p != 1)

    # ── Phase 1: 训练一个最小模型作为 eval 目标 ──────────────────────────
    _log("── Phase 1: Train minimal model ──")
    from ai.train.args import get_config as make_train_args
    train_parser = make_train_args()
    train_args = train_parser.parse_args(["--scenario", args.scenario])
    # 不传 --use-gnn 的话默认 False, 但 args.use_gnn 可能被用户设了
    if args.use_gnn:
        train_args.use_gnn = True

    from ai.train.sb3_trainer import Sb3Trainer
    trainer = Sb3Trainer(train_args)
    trainer.agent._model.learn(total_timesteps=2048)
    step = trainer.agent.num_timesteps
    _log(f"  trained {step} steps, saving checkpoint...")

    ckpt_dir = os.path.join(trainer.save_dir, "smoke_eval")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = os.path.join(ckpt_dir, f"ckpt_{step}")
    trainer.agent.save(ckpt)
    trainer.env.close()
    _log(f"  ✓ checkpoint saved to {ckpt}.zip")

    # ── Phase 2: 跑 evaluate() ────────────────────────────────────────────
    _log("\n── Phase 2: Run evaluate() ──")

    opp_types = [s.strip() for s in args.eval_opponent.split(",")]
    specs = []
    for t in opp_types:
        specs.append({"type": t, "player_id": opponent_id})

    def _make_env(i: int):
        def _init():
            env = LwgEnv(args.scenario)
            env.set_opponent(specs[i % len(specs)])
            return env
        return _init

    n = args.n_envs
    venv = VecMonitor(
        SubprocVecEnv([_make_env(i) for i in range(n)]),
        info_keywords=("win", "turn"),
    )

    results = evaluate(ckpt, venv, args.eval_episodes, specs)
    _log(f"  ✓ {len(results)} results")
    for r in results:
        _log(f"    vs={r.opponent_spec['type']:8s}  "
             f"win={r.wins}/{r.episodes} ({r.win_rate:.2f})  "
             f"turns={r.avg_turns:.1f}" if r.avg_turns else f"turns=None")

    venv.close()

    # ── 校验 ──────────────────────────────────────────────────────────────
    _log("\n── Checks ──")

    if len(results) != n:
        _fail(f"expected {n} results, got {len(results)}")

    total_eps = sum(r.episodes for r in results)
    if total_eps != n * args.eval_episodes:
        _fail(f"total episodes: {total_eps} != {n} * {args.eval_episodes}")

    for r in results:
        if r.episodes == 0:
            _fail("zero episodes in result")
        if r.win_rate < 0 or r.win_rate > 1:
            _fail(f"win_rate out of range: {r.win_rate}")
        if r.avg_turns is not None and r.avg_turns <= 0:
            _fail(f"avg_turns <= 0: {r.avg_turns}")

    wr = aggregate_win_rate(results)
    at = aggregate_avg_turns(results)
    if wr is None or wr < 0 or wr > 1:
        _fail(f"aggregate win_rate: {wr}")
    if at is not None and at <= 0:
        _fail(f"aggregate avg_turns: {at}")
    _log(f"  aggregate: win_rate={wr:.2f}  avg_turns={at:.1f}" if at else f"  aggregate: win_rate={wr:.2f}")

    # ── 判定 ──────────────────────────────────────────────────────────────
    _log(f"\n{'='*60}")
    if _VERDICT:
        _log(f"EVAL SMOKE TEST FAILED ({len(_VERDICT)} issues)")
        for v in _VERDICT:
            _log(f"  - {v}")
        _log(f"{'='*60}")
        sys.exit(1)
    else:
        _log("EVAL SMOKE TEST PASSED")
        _log(f"{'='*60}")

    _stop_log()


def _log_path(args) -> str:
    from ai.train.utils import resolve_save_dir
    d = resolve_save_dir(args.scenario, args.save_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(d, f"eval_smoke_{ts}.log")


if __name__ == "__main__":
    main()
