"""自博弈训练编排：chunked model.learn() + 策略池动态换对手。

每个 chunk 结束后保存 checkpoint 并加入策略池，后续 chunk 的对手从池中
选取（SP 模式始终取最新 checkpoint）。
"""

from __future__ import annotations

import os

from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecMonitor

from ai.algos.opponent_pool import OpponentPool
from ai.algos.policy import SB3Policy
from ai.envs.env import LwgEnv
from ai.envs.opponents import PolicyOpponent, RandomOpponent, RuleOpponent
from ai.envs.opponents.base_opponent import BaseOpponent
from ai.train.metrics import WinRateCallback


def _make_warmup_opponent(opponent_type: str, player_id: int) -> BaseOpponent:
    if opponent_type == "random":
        return RandomOpponent(player_id=player_id)
    if opponent_type == "rule":
        return RuleOpponent(player_id=player_id)
    raise ValueError(f"Unknown initial opponent type: {opponent_type!r}")


def train_self_play(args) -> None:
    from ai.train.sb3_trainer import _resolve_save_dir, _set_seeds

    save_dir = _resolve_save_dir(args)
    os.makedirs(save_dir, exist_ok=True)
    tb_log_dir = os.path.join(save_dir, "tb")
    _set_seeds(args.seed)

    env = VecMonitor(
        make_vec_env(lambda: LwgEnv(args.scenario), n_envs=1),
        info_keywords=("win", "turn"),
    )

    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs={"net_arch": args.net_arch},
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        verbose=1,
        seed=args.seed,
        tensorboard_log=tb_log_dir,
    )

    # --- logging ---
    project = args.wandb_project or args.scenario.split("/")[0]
    exp_name = args.exp_name or args.scenario.split("/")[-1]

    callbacks = [WinRateCallback(window=args.win_rate_window)]

    if args.wandb:
        import wandb
        from wandb.integration.sb3 import WandbCallback

        wandb.init(
            project=project,
            name=exp_name,
            config=vars(args),
            dir=save_dir,
            sync_tensorboard=True,
            monitor_gym=True,
        )
        callbacks.append(WandbCallback(verbose=2))
    else:
        model.set_logger(configure(folder=None, format_strings=["stdout"]))

    # --- self-play setup ---
    pool = OpponentPool(max_size=args.self_play_pool_size)

    agent_id: int = env.get_attr("agent_id")[0]                                       # type: ignore[assignment]
    num_players: int = env.get_attr("config")[0].game.num_players                      # type: ignore[assignment]
    opponent_id = next(p for p in range(1, num_players + 1) if p != agent_id)

    obs_enc = env.get_attr("obs_encoder")[0]
    act_enc = env.get_attr("act_encoder")[0]

    # 冷启动：第一个 chunk 用固定对手
    warmup_opp = _make_warmup_opponent(args.self_play_initial_opponent, opponent_id)
    env.env_method("set_opponent", warmup_opp)
    print(f"[SelfPlay] 冷启动对手: {args.self_play_initial_opponent}")

    # --- chunked training ---
    total = args.total_timesteps
    chunk = args.checkpoint_freq
    num_chunks = total // chunk

    print(f"[SelfPlay] 总步数={total}, chunk={chunk}, 共{num_chunks}个chunk, 池容量={args.self_play_pool_size}")

    for i in range(num_chunks):
        step = (i + 1) * chunk
        reset_num_timesteps = (i == 0)
        model.learn(chunk, reset_num_timesteps=reset_num_timesteps, callback=callbacks)

        # 保存 checkpoint 并加入策略池
        ckpt_path = os.path.join(save_dir, f"lwg_ppo_{step}_steps.zip")
        model.save(ckpt_path)
        evicted = pool.add(ckpt_path, step)
        if evicted is not None:
            print(f"[SelfPlay] 池满，淘汰: {evicted.path} (step={evicted.step})")

        # 切换对手为池中最新的 checkpoint
        entry = pool.latest()
        if entry is not None:
            new_opp = PolicyOpponent(
                player_id=opponent_id,
                policy=SB3Policy(entry.path),
                obs_encoder=obs_enc,
                act_encoder=act_enc,
            )
            env.env_method("set_opponent", new_opp)
            print(f"[SelfPlay] chunk {i+1}/{num_chunks} 完成, 切换对手: step={entry.step}")

    model.save(os.path.join(save_dir, "final"))
    print(f"模型已保存至 {save_dir}/final.zip")
