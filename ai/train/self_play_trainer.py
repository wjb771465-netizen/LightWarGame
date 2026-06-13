from __future__ import annotations

import logging
import os

from ai.algos.opponent_pool import OpponentPool
from ai.algos.policy import SB3Policy
from ai.train.sb3_trainer import Sb3Trainer


class SelfPlayTrainer(Sb3Trainer):

    # ------------------------------------------------------------------
    # Opponent sampling
    # ------------------------------------------------------------------

    def _sample_opponent_specs(self, pool, n_total: int, opponent_id: int) -> list[dict]:
        """从 OpponentPool 中有放回采样 n_total 个对手 spec。"""
        strategy = self.args.pool_sampling_strategy
        lam = self.args.sampling_lam
        scale = self.args.sampling_scale
        ft = self.args.self_play_initial_opponent

        specs = []
        for _ in range(n_total):
            if strategy == "uniform":
                entry = pool.sample_uniform()
            elif strategy == "progress":
                entry = pool.sample_progress(lam=lam, s=scale, D=self.args.progress_D)
            elif strategy == "elo":
                entry = pool.sample_elo(lam=lam, s=scale)
            else:  # latest
                entry = pool.latest()

            if entry is not None:
                specs.append({
                    "type": "policy", "player_id": opponent_id,
                    "path": entry.path.replace(".zip", ""),
                })
            else:
                specs.append({"type": ft, "player_id": opponent_id})

        return specs

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, agent: SB3Policy, env) -> None:
        agent_id: int = env.get_attr("agent_id")[0]
        num_players: int = env.get_attr("config")[0].game.num_players
        opponent_id = next(p for p in range(1, num_players + 1) if p != agent_id)

        # 对手种类数，--n-envs 须为其整数倍
        n_opponents = self.args.n_opponents or self.args.n_envs
        n_envs = self.args.n_envs
        if n_envs % n_opponents != 0:
            raise ValueError(
                f"--n-envs ({n_envs}) 须为 --n-opponents ({n_opponents}) 的整数倍"
            )
        per_opponent = n_envs // n_opponents  # 每个对手分给几个 env

        self._pool = OpponentPool(max_size=self.args.self_play_pool_size)

        warmup_spec = {"type": self.args.self_play_initial_opponent, "player_id": opponent_id}
        env.env_method("set_opponent", warmup_spec)
        logging.info("[SelfPlay] 冷启动对手: %s", self.args.self_play_initial_opponent)

        agent_elo = 1200.0
        total = self.args.total_timesteps
        chunk = self.args.checkpoint_freq
        while agent.num_timesteps < total:
            agent.learn(min(chunk, total - agent.num_timesteps), callback=[self._win_cb])
            agent._model._custom_logger = True
            step = agent.num_timesteps
            ckpt = os.path.join(self.save_dir, f"ckpt_{step}")
            agent.save(ckpt)

            ckpt_zip = ckpt + ".zip"
            if self.args.use_eval:
                results = self.eval(ckpt, env, step)
                prev_elo = agent_elo
                evicted, agent_elo, accepted = self._pool.add(
                    ckpt_zip, step, elo=agent_elo, outcomes=results)
                if accepted:
                    if evicted is not None:
                        logging.info("[SelfPlay] 池满，淘汰: %s (step=%d)", evicted.path, evicted.step)
                    logging.info("[SelfPlay] step=%d, ELO %.1f -> %.1f, 入池",
                                 step, prev_elo, agent_elo)
                else:
                    logging.info("[SelfPlay] step=%d, ELO %.1f -> %.1f, 跳过入池",
                                 step, prev_elo, agent_elo)
            else:
                evicted, agent_elo, _ = self._pool.add(ckpt_zip, step)
                if evicted is not None:
                    logging.info("[SelfPlay] 池满，淘汰: %s (step=%d)", evicted.path, evicted.step)

            self.log_eval_metrics({"elo": agent_elo}, step)

            # 采样 n_opponents 种对手，每种发给 per_opponent 个 env
            specs = self._sample_opponent_specs(self._pool, n_opponents, opponent_id)
            steps = []
            for i, spec in enumerate(specs):
                indices = list(range(i * per_opponent, (i + 1) * per_opponent))
                env.env_method("set_opponent", spec, indices=indices)
                if spec["type"] == "policy":
                    steps.append(int(spec["path"].rsplit("ckpt_", 1)[-1]))
                else:
                    steps.append(spec["type"])
            steps_str = ", ".join(str(s) for s in steps[:8])
            if len(steps) > 8:
                steps_str += ", ..."
            logging.info("[SelfPlay] step=%d, opponents=[%s], envs=%d, per=%d",
                         step, steps_str, n_envs, per_opponent)

        agent.save(os.path.join(self.save_dir, "final"))
        logging.info("模型已保存至 %s/final.zip", self.save_dir)
        self.render(os.path.join(self.save_dir, "final"))

    # ------------------------------------------------------------------
    # Eval opponents
    # ------------------------------------------------------------------

    def choose_eval_opponents(self, env, include_fixed: bool = True, region: int | None = None) -> list[dict]:
        """从 OpponentPool 采样 eval 对手 + 可选固定对手。"""
        eval_n_envs = self.args.eval_n_envs or self.args.n_envs
        eval_n_opponents = self.args.eval_n_opponents or self.args.n_opponents or self.args.n_envs
        opponent_id = self._opponent_id(env)

        if len(self._pool) == 0:
            pool_specs = eval_n_envs * [
                {"type": self.args.self_play_initial_opponent, "player_id": opponent_id},
            ]
        else:
            per_opponent = max(1, eval_n_envs // eval_n_opponents)
            specs = self._sample_opponent_specs(self._pool, eval_n_opponents, opponent_id)
            pool_specs = []
            for spec in specs:
                pool_specs.extend([spec] * per_opponent)
            pool_specs = pool_specs[:eval_n_envs]

        if not (include_fixed and self.args.eval_opponent):
            return pool_specs

        # 固定对手与池对手各占一半 env
        half = max(1, eval_n_envs // 2)
        pool_specs = pool_specs[:half]
        fixed_specs = self._fixed_opponent_specs(eval_n_envs - half, opponent_id)
        return pool_specs + fixed_specs
