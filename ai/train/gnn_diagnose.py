"""GNN 训练诊断：打印参数变化、梯度流、特征统计到日志文件。

用法:
  conda run -n chinese_war_game python -m ai.train.gnn_diagnose \
    --scenario duel/vsbaseline_gnn_fixed --use-gnn --gnn-hidden-channels 128 \
    --net-arch 128 --total-timesteps 16384 --n-steps 2048 --batch-size 512 \
    --n-epochs 10 --lr 3e-4 --gamma 0.99 --gae-lambda 0.97 --clip-range 0.2 \
    --seed 42

输出: ai/train/results/<scenario>/gnn_diag_<timestamp>.log
"""

from __future__ import annotations

import io
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback

_LOG_STREAM: io.StringIO | None = None
_LOG_PATH: str | None = None


# ── helpers ──────────────────────────────────────────────────────────────

def _log(msg: str = "") -> None:
    """同时输出到 stdout 和日志文件。"""
    print(msg, flush=True)
    if _LOG_STREAM is not None:
        _LOG_STREAM.write(msg + "\n")


def _stop_log() -> None:
    """停止记录，将缓冲区写入文件。"""
    global _LOG_STREAM, _LOG_PATH
    if _LOG_STREAM is not None and _LOG_PATH is not None:
        with open(_LOG_PATH, "a") as f:
            f.write(_LOG_STREAM.getvalue())
    _LOG_STREAM = None


def _start_log(filepath: str) -> None:
    """开始双写：终端 + 文件缓冲。"""
    global _LOG_STREAM, _LOG_PATH
    _LOG_STREAM = io.StringIO()
    _LOG_PATH = filepath
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # 新建文件，写入头部
    with open(filepath, "w") as f:
        f.write(f"# GNN diagnose log — {datetime.now().isoformat()}\n")
        f.write(f"# PID={os.getpid()}\n\n")


def _norm(x: torch.Tensor | nn.Parameter) -> float:
    return float(x.data.norm().item())


def _grad_norm(p: nn.Parameter) -> float:
    if p.grad is None:
        return 0.0
    return float(p.grad.data.norm().item())


def _flat_grad_norm(params: list[nn.Parameter]) -> float:
    s = 0.0
    for p in params:
        if p.grad is not None:
            s += float(p.grad.data.norm().item()) ** 2
    return s**0.5


# ── parameter tree ──────────────────────────────────────────────────────

def _param_tree(module: nn.Module) -> dict[str, list[nn.Parameter]]:
    """按前缀分组：features_extractor.backbone.* / mlp_extractor.* / action_net / value_net"""
    groups: dict[str, list[nn.Parameter]] = {}
    for name, p in module.named_parameters():
        if name.startswith("features_extractor.backbone."):
            key = "gnn." + name.removeprefix("features_extractor.backbone.")
        elif name.startswith("features_extractor."):
            key = "extractor." + name.removeprefix("features_extractor.")
        elif name.startswith("mlp_extractor."):
            key = "mlp." + name.removeprefix("mlp_extractor.")
        elif name.startswith("action_net."):
            key = "action." + name.removeprefix("action_net.")
        elif name.startswith("value_net."):
            key = "value." + name.removeprefix("value_net.")
        else:
            key = "other." + name
        groups.setdefault(key, []).append(p)
    return groups


def _snapshot(params: dict[str, list[nn.Parameter]]) -> dict[str, float]:
    """参数当前 L2 范数。"""
    return {k: sum(_norm(p) for p in plist) for k, plist in params.items()}


def _delta(
    before: dict[str, float], after: dict[str, float]
) -> dict[str, float]:
    return {k: abs(after[k] - before[k]) for k in before}


def _grad_snapshot(params: dict[str, list[nn.Parameter]]) -> dict[str, float]:
    return {k: _flat_grad_norm(plist) for k, plist in params.items()}


# ── forward hooks ───────────────────────────────────────────────────────

_FWD_STATS: dict[str, list[float]] = {}


def _forward_hook(name: str):
    def hook(_module, _input, output):
        if isinstance(output, torch.Tensor):
            _FWD_STATS.setdefault(name, []).append(
                float(output.detach().std().item())
            )
    return hook


def _install_fwd_hooks(module: nn.Module) -> list:
    handles = []
    for n, m in module.named_modules():
        if n == "":
            continue
        # 只在关键层挂 hook
        if any(k in n for k in ("conv1", "conv2", "norm1", "norm2", "head", "res_proj",
                                  "action_net", "value_net")):
            handles.append(m.register_forward_hook(_forward_hook(n)))
    return handles


# ── callback ─────────────────────────────────────────────────────────────

_UPDATE_COUNT = 0
_REF_SNAPSHOT: dict[str, float] | None = None


def _set_ref_snapshot(model):
    """在第一个 rollout 开始前保存参数快照。"""
    global _REF_SNAPSHOT
    _REF_SNAPSHOT = _snapshot(_param_tree(model.policy))


class GnnDiagnoseCallback(BaseCallback):
    """每个 rollout 结束后打印诊断信息。"""

    def __init__(self, log_interval: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval

    def _on_training_start(self) -> None:
        _log("=" * 70)
        _log("GNN Diagnose — training start")
        _log("=" * 70)
        self._print_model_info()

    def _on_rollout_end(self) -> None:
        global _UPDATE_COUNT, _REF_SNAPSHOT

        _UPDATE_COUNT += 1
        if _UPDATE_COUNT % self.log_interval != 0:
            return

        model = self.model
        params = _param_tree(model.policy)
        cur = _snapshot(params)

        _log(f"\n{'─'*60}")
        _log(f"[update {_UPDATE_COUNT}] {datetime.now().strftime('%H:%M:%S')}")

        # 1. 参数变化 vs 参考快照
        if _REF_SNAPSHOT is not None:
            d = _delta(_REF_SNAPSHOT, cur)
            _log("  param Δ vs init:")
            for k in sorted(d):
                v = d[k]
                marker = " *** ZERO" if v < 1e-12 else ""
                _log(f"    {k:40s} Δ={v:.6e}{marker}")

        # 2. 参数绝对值
        _log("  param norm:")
        for k in sorted(cur):
            _log(f"    {k:40s} norm={cur[k]:.6e}")

        # 3. 梯度范数（最近一次 backward 后的残留）
        grads = _grad_snapshot(params)
        any_grad = any(v > 0 for v in grads.values())
        _log(f"  grad norm (last backward):")
        for k in sorted(grads):
            _log(f"    {k:40s} |g|={grads[k]:.6e}")
        if not any_grad:
            _log("  ⚠️  ALL GRADIENTS ARE ZERO — policy is not learning!")

    def _on_training_end(self) -> None:
        _log(f"\n{'='*70}")
        _log("GNN Diagnose — training end")
        _log(f"total updates: {_UPDATE_COUNT}")
        _log("=" * 70)

    def _print_model_info(self) -> None:
        model = self.model
        policy = model.policy

        _log("\n── Model structure ──")
        _log(f"policy type: {type(policy).__name__}")
        _log(f"features_extractor: {type(policy.features_extractor).__name__}")
        _log(f"observation_space: {model.observation_space}")
        _log(f"action_space: {model.action_space}")

        _log("\n── Parameter requires_grad ──")
        params = _param_tree(policy)
        for k, plist in sorted(params.items()):
            all_grad = all(p.requires_grad for p in plist)
            n = len(plist)
            shapes = [tuple(p.shape) for p in plist]
            flag = "✓" if all_grad else "✗ NO_GRAD"
            _log(f"  {k:40s} n={n} requires_grad={flag} shapes={shapes}")

        _log("\n── Optimizer param_groups ──")
        if hasattr(model, "_optimizer") and model._optimizer is not None:
            opt = model._optimizer
        elif hasattr(model, "policy") and hasattr(model.policy, "optimizer"):
            opt = model.policy.optimizer
        else:
            opt = None
            _log("  (optimizer not accessible from callback)")

        if opt is not None:
            for i, pg in enumerate(opt.param_groups):
                n = len(pg["params"])
                lr = pg.get("lr", "?")
                _log(f"  group[{i}]: {n} params, lr={lr}")
                total_el = sum(p.numel() for p in pg["params"])
                _log(f"    total elements: {total_el}")


# ── backward hook for gradient capture ───────────────────────────────────

_GRAD_STATS: dict[str, list[float]] = {}
_GRAD_HANDLES: list = []


def _backward_hook(name: str):
    def hook(grad):
        if grad is not None:
            _GRAD_STATS.setdefault(name, []).append(float(grad.norm().item()))
    return hook


def _install_bwd_hooks(module: nn.Module):
    global _GRAD_HANDLES
    for n, p in module.named_parameters():
        if not p.requires_grad:
            continue
        h = p.register_hook(_backward_hook(n))
        _GRAD_HANDLES.append(h)


def _remove_bwd_hooks():
    global _GRAD_HANDLES
    for h in _GRAD_HANDLES:
        h.remove()
    _GRAD_HANDLES.clear()


# ── activation diagnosis ─────────────────────────────────────────────────

def diagnose_activations(model, env, n_steps: int = 256) -> None:
    """前向传播诊断：检查各层激活值方差、logits 分布。"""
    _log(f"\n── Forward activation stats ({n_steps} steps) ──")
    global _FWD_STATS
    _FWD_STATS.clear()

    policy = model.policy
    handles = _install_fwd_hooks(policy)

    obs = env.reset()
    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=False)
        obs, _, dones, _ = env.step(action)
        if np.any(dones):
            obs = env.reset()

    for h in handles:
        h.remove()

    for name, values in sorted(_FWD_STATS.items()):
        arr = np.array(values)
        _log(f"  {name:45s} mean_std={arr.mean():.4f}  min_std={arr.min():.4f}  max_std={arr.max():.4f}")

    # Check for zero-variance activations (dead layers)
    dead = [n for n, v in _FWD_STATS.items() if np.array(v).mean() < 1e-8]
    if dead:
        _log(f"  ⚠️  ZERO-VARIANCE layers: {dead}")
    else:
        _log("  ✓ all layers have non-zero output variance")


def diagnose_logits(model, env, n_steps: int = 512) -> None:
    """检查 action logits 是否变化、entropy 是否正常。"""
    _log(f"\n── Logits / entropy diagnosis ({n_steps} steps) ──")
    policy = model.policy
    logits_list = []
    masks_list = []

    obs = env.reset()
    for _ in range(n_steps):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=policy.device)
            features = policy.extract_features(obs_t)
            latent = policy.mlp_extractor.forward_actor(features)
            logits = policy.action_net(latent)
        logits_list.append(logits.cpu().numpy())

        mask = env.env_method("action_masks")
        masks_list.append(np.array(mask))

        action, _ = model.predict(obs, deterministic=False)
        obs, _, dones, _ = env.step(action)
        if np.any(dones):
            obs = env.reset()

    logits_all = np.concatenate(logits_list, axis=0)  # (total_steps, action_dim)

    _log(f"  logits shape: {logits_all.shape}")
    _log(f"  logits mean: {logits_all.mean():.4f}")
    _log(f"  logits std (across actions): {logits_all.std(axis=1).mean():.4f}")
    _log(f"  logits std (across samples): {logits_all.std(axis=0).mean():.4f}")

    # Per-sample logits variation
    sample_std = logits_all.std(axis=1)  # std across actions per sample
    _log(f"  per-sample action-std:  mean={sample_std.mean():.4f}  min={sample_std.min():.4f}  max={sample_std.max():.4f}")
    if sample_std.mean() < 1e-6:
        _log("  ⚠️  ALL SAMPLES HAVE NEARLY IDENTICAL LOGITS — action_net output is collapsed!")

    # Pairwise comparison: are consecutive logits different?
    if len(logits_all) > 1:
        diffs = np.abs(logits_all[1:] - logits_all[:-1]).mean(axis=1)
        _log(f"  consecutive logits |diff|: mean={diffs.mean():.6f}  max={diffs.max():.6f}")

    # Entropy of softmax(logits)
    logits_t = torch.as_tensor(logits_all, dtype=torch.float32)
    probs = torch.softmax(logits_t, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
    max_entropy = np.log(logits_all.shape[-1])
    _log(f"  entropy: mean={entropy.mean():.4f}  (max possible={max_entropy:.4f})")

    # Mask statistics
    masks_all = np.concatenate(masks_list, axis=0)
    masked_count = (masks_all == 0).sum(axis=1)
    total_actions = masks_all.shape[1]
    _log(f"  mask: avg masked actions = {masked_count.mean():.1f} / {total_actions}")
    _log(f"        only no-op available = {(masked_count == total_actions - 1).mean()*100:.1f}%")


# ── main ─────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    from ai.train.args import get_config
    from ai.envs.env import LwgEnv
    from ai.envs.utils import parse_config
    from ai.algos.extractors import GNNExtractor
    from ai.algos.gnn import adj_to_edge_index
    from ai.algos.policy import SB3Policy
    from ai.train.utils import resolve_save_dir

    args = get_config().parse_args()
    if not args.use_gnn:
        _log("ERROR: --use-gnn is required for this diagnostic")
        sys.exit(1)

    # 日志路径
    scenario_dir = resolve_save_dir(args.scenario, args.save_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(scenario_dir, f"gnn_diag_{ts}.log")
    _start_log(log_path)
    _log(f"Log: {log_path}")

    # 构建环境
    _log("Building env...")
    cfg = parse_config(args.scenario)
    env = LwgEnv(args.scenario)

    # 构建模型
    game_map = env.game_map
    obs_encoder = env.obs_encoder
    edge_index = adj_to_edge_index(game_map.adjacency_matrix)

    policy_kwargs = dict(
        features_extractor_class=GNNExtractor,
        features_extractor_kwargs=dict(
            num_regions=len(game_map.regions) - 1,
            feat_dim=obs_encoder._F,
            global_dim=obs_encoder._G,
            edge_index=edge_index,
            hidden_channels=args.gnn_hidden_channels,
        ),
        net_arch=args.net_arch,
    )

    _log("\n── Policy kwargs ──")
    for k, v in policy_kwargs.items():
        if k == "features_extractor_kwargs":
            for kk, vv in v.items():
                if kk != "edge_index":
                    _log(f"  {kk}: {vv}")
        else:
            _log(f"  {k}: {v}")

    _log("\n── Building MaskablePPO with GNNExtractor ──")
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv

    def _mk_env():
        e = LwgEnv(args.scenario)
        return e

    venv = make_vec_env(_mk_env, n_envs=1, vec_env_cls=SubprocVecEnv)

    model_cls = None
    try:
        from sb3_contrib import MaskablePPO
        model_cls = MaskablePPO
    except ImportError:
        from stable_baselines3 import PPO
        model_cls = PPO
        _log("WARNING: sb3_contrib not found, using vanilla PPO (no action masking)")

    model = model_cls(
        "MlpPolicy", venv,
        policy_kwargs=policy_kwargs,
        n_steps=getattr(args, "n_steps", 2048),
        batch_size=getattr(args, "batch_size", 512),
        n_epochs=getattr(args, "n_epochs", 10),
        learning_rate=getattr(args, "lr", 3e-4),
        gamma=getattr(args, "gamma", 0.99),
        gae_lambda=getattr(args, "gae_lambda", 0.97),
        clip_range=getattr(args, "clip_range", 0.2),
        verbose=1,
        seed=getattr(args, "seed", 42),
    )

    _set_ref_snapshot(model)

    # ── Phase 1: One-time checks ──
    _log("\n── 1. Optimizer check ──")
    try:
        opt = model.policy.optimizer
    except AttributeError:
        _log("  model.policy.optimizer not found")
        opt = None
    if opt is not None:
        opt_params = set()
        for pg in opt.param_groups:
            for p in pg["params"]:
                opt_params.add(id(p))
        all_params = [p for p in model.policy.parameters()]
        missing = [p for p in all_params if id(p) not in opt_params]
        if missing:
            _log(f"  ⚠️  {len(missing)}/{len(all_params)} params NOT in optimizer!")
            for p in missing[:5]:
                _log(f"    shape={tuple(p.shape)} requires_grad={p.requires_grad}")
        else:
            _log(f"  ✓ all {len(all_params)} params in optimizer")

        # Check GNN params specifically
        gnn_params = [p for n, p in model.policy.named_parameters() if n.startswith("features_extractor.backbone.")]
        gnn_in_opt = [p for p in gnn_params if id(p) in opt_params]
        _log(f"  GNN backbone: {len(gnn_in_opt)}/{len(gnn_params)} params in optimizer")
    else:
        _log("  SKIP (no access to optimizer)")

    # ── Phase 2: Forward diagnosis ──
    diagnose_activations(model, env)
    diagnose_logits(model, env)

    # ── Phase 3: Short training with gradient capture ──
    total = getattr(args, "total_timesteps", 4096)
    _log(f"\n── 2. Training {total} steps with gradient hooks ──")

    _install_bwd_hooks(model.policy)
    callback = GnnDiagnoseCallback(log_interval=1)
    model.learn(total_timesteps=total, callback=callback, reset_num_timesteps=True)

    # Print collected backward grad stats
    if _GRAD_STATS:
        _log(f"\n── Backward gradient stats (collected during training) ──")
        for name, values in sorted(_GRAD_STATS.items()):
            arr = np.array(values)
            _log(f"  {name:50s} n={len(arr):4d}  mean|g|={arr.mean():.6e}  last|g|={arr[-1]:.6e}")
        # Flag zero-gradient params
        zero_grads = [n for n, v in _GRAD_STATS.items() if np.array(v).mean() < 1e-15]
        if zero_grads:
            _log(f"\n  ⚠️  ZERO GRADIENT params ({len(zero_grads)}):")
            for n in zero_grads[:10]:
                _log(f"    {n}")
        else:
            _log("  ✓ all params received non-zero gradients at some point")

    _remove_bwd_hooks()

    # ── Phase 4: Final param check ──
    _log(f"\n── 3. Final parameter Δ ──")
    params = _param_tree(model.policy)
    cur = _snapshot(params)
    if _REF_SNAPSHOT is not None:
        d = _delta(_REF_SNAPSHOT, cur)
        _log("  Δ vs init:")
        unchanged = []
        for k in sorted(d):
            v = d[k]
            msg = f"    {k:40s} Δ={v:.6e}"
            if v < 1e-12:
                msg += " *** ZERO"
                unchanged.append(k)
            _log(msg)
        if unchanged:
            _log(f"\n  ⚠️  {len(unchanged)} groups had ZERO change — policy is frozen!")
        else:
            _log("  ✓ all param groups changed")
    else:
        _log("  (no ref snapshot)")

    _log(f"\nLog saved to: {log_path}")
    _stop_log()
    env.close()
    venv.close()


if __name__ == "__main__":
    main()
