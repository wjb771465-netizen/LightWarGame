from __future__ import annotations

import os

from ai.algos.opponent_pool import OpponentPool
from ai.algos.policy import SB3Policy
from ai.envs.opponents import PolicyOpponent, RandomOpponent, RuleOpponent
from ai.envs.opponents.base_opponent import BaseOpponent
from ai.train.sb3_trainer import Sb3Trainer


class SelfPlayTrainer(Sb3Trainer):

    @staticmethod
    def _make_warmup_opponent(opponent_type: str, player_id: int) -> BaseOpponent:
        if opponent_type == "random":
            return RandomOpponent(player_id=player_id)
        if opponent_type == "rule":
            return RuleOpponent(player_id=player_id)
        raise ValueError(f"Unknown initial opponent type: {opponent_type!r}")

    def run(self, agent: SB3Policy, env) -> None:
        agent_id: int = env.get_attr("agent_id")[0]
        num_players: int = env.get_attr("config")[0].game.num_players
        opponent_id = next(p for p in range(1, num_players + 1) if p != agent_id)
        obs_enc = env.get_attr("obs_encoder")[0]
        act_enc = env.get_attr("act_encoder")[0]

        pool = OpponentPool(max_size=self.args.self_play_pool_size)
        env.env_method("set_opponent", self._make_warmup_opponent(self.args.self_play_initial_opponent, opponent_id))
        print(f"[SelfPlay] 冷启动对手: {self.args.self_play_initial_opponent}")

        total = self.args.total_timesteps
        chunk = self.args.checkpoint_freq
        while agent.num_timesteps < total:
            agent.learn(min(chunk, total - agent.num_timesteps), callback=[self._win_cb])
            step = agent.num_timesteps
            ckpt = os.path.join(self.save_dir, f"ckpt_{step}")
            agent.save(ckpt)
            self.log_metrics(self._collect_metrics(), step)

            evicted = pool.add(ckpt + ".zip", step)
            if evicted is not None:
                print(f"[SelfPlay] 池满，淘汰: {evicted.path} (step={evicted.step})")

            strategy = self.args.pool_sampling_strategy
            lam = self.args.sampling_lam
            scale = self.args.sampling_scale
            if strategy == "uniform":
                entry = pool.sample_uniform()
            elif strategy == "progress":
                entry = pool.sample_progress(lam=lam, s=scale, D=self.args.progress_D)
            elif strategy == "elo":
                entry = pool.sample_elo(lam=lam, s=scale)
            else:  # "latest"
                entry = pool.latest()

            if entry is not None:
                env.env_method("set_opponent", PolicyOpponent(
                    player_id=opponent_id,
                    policy=SB3Policy(path=entry.path),
                    obs_encoder=obs_enc,
                    act_encoder=act_enc,
                ))
                print(f"[SelfPlay] step={step}, strategy={strategy}, 对手: {entry.path} (step={entry.step})")

        agent.save(os.path.join(self.save_dir, "final"))
        print(f"模型已保存至 {self.save_dir}/final.zip")
