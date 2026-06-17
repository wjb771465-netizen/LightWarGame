# 训练崩溃分析：Critic 过拟合 → 策略死亡螺旋

> 分析对象：W&B runs `zpbodjhq`（异常）vs `pdpwlkeg`（正常），均为 RegionSelfPlay PFSP，region 4 & 20。

## 一、现象

异常运行在 3.3M steps 后，两个 region 同时崩溃：

| 指标 | 正常运行 | 异常运行 |
|------|---------|---------|
| 最终 ELO | **3142** | **749** |
| ELO 峰值 | 持续上升 | 1746 @ 3.27M |
| vs_random 终局胜率 | 1.00 | 0.00 |
| vs_rule 终局胜率 | 0.75 | 0.00 |
| vs_fsm 终局胜率 | (未评估) | 始终 0.00 |

**关键：这不是评估假象，而是训练本身崩溃了。** 异常运行的 rollout win_rate 在训练中就跌到 30%，训练奖励出现大量负值。

## 二、崩溃的因果链

### 阶段 1：过早收敛（1M-2.5M）

表面上异常运行的训练数据甚至更好（win_rate 0.83 vs 0.76），但暗藏危机：

| 指标 | 异常 | 正常 | 差异 |
|------|------|------|------|
| win_rate_200 | 0.83 | 0.76 | **+9%** |
| value_loss | 844 | 591 | **+43%** ⚠️ |
| entropy_loss | -2.33 | -2.71 | **更确定性** ⚠️ |
| approx_kl | 0.0053 | 0.0065 | **更新量更小** ⚠️ |

策略找到了一个能打败当前池对手的"舒适策略"，但探索太少（高 entropy_loss = 低熵 = 更确定性）。这导致 rollout 数据缺乏多样性，价值函数只见过狭窄的对手分布。

### 阶段 2：价值函数崩盘（2.5M-3.0M）

池中不断加入新 checkpoint，对手分布发生变化。过拟合的 Critic 无法泛化到新对手：

```
value_loss 比值（异常/正常）：
  @2.5M: 1480 / 579 = 2.56x
  @3.0M: 1360 / 266 = 5.11x
```

### 阶段 3：策略停止学习（~3.0M）

**价值函数是 PPO 的"方向盘"。** Critic 出错 → 优势估计 A(s,a) 出错 → 策略梯度信号崩溃：

```
               异常        正常
PG_loss        -0.00      -0.01    ← 策略梯度几乎为零
clip_fraction   0.030      0.062   ← 大部分更新碰不到 clip 边界
approx_kl       0.0036     0.0072  ← 策略几乎不变化
```

此时训练已名存实亡——策略不再更新，不再探索，只是机械地重复（越来越差的）动作。

### 阶段 4：死亡螺旋（3.0M-3.5M）

```
win_rate_200:  0.83 → 0.63 → 0.46 → 0.37 → 0.30
ep_rew_mean:    83 →   36 →  -25 →   -6 →  -39
entropy_loss: -2.33 → -1.16 → -1.07 → -0.83
```

agent 在**训练中就开始输**。负奖励进一步毒化价值函数，形成不可逆的正反馈循环。

### 阶段 5：无法自救（3.5M-4.85M）

entropy 彻底锁死在 -0.83 ~ -1.0，策略 100% 确定性，无法产生任何探索行为。对固定对手和池对手的胜率同时归零。

## 三、根因总结

```
策略过早收敛（entropy 低，exploration 不足）
  → rollout 数据缺乏多样性
    → Critic 过拟合到狭窄的对手分布
      → 对手分布变化时 Critic 预测失准
        → 优势估计错误 → 策略梯度消失
          → 策略不再更新 → entropy 进一步下降
            → 无法探索 → 彻底崩溃
```

**本质上是 Critic 过拟合触发的策略死亡螺旋。** Actor 和 Critic 互相锁死：Critic 给出错误信号让 Actor 变差，Actor 变差产生更差的数据让 Critic 更错。

## 四、调参建议

### 高优先级：阻止 entropy 过早崩溃

| 参数 | 当前值 | 建议值 | 理由 |
|------|--------|--------|------|
| `ent_coef`（PPO 熵正则系数） | 默认 0.0 | **0.01 ~ 0.05** | 直接惩罚低熵策略，强制保留探索噪声。这是最关键的改动 |
| `--n-epochs` | 10 | **5 ~ 8** | 减少每批数据的重复训练轮数，降低对当前分布的过拟合速度 |

具体改动（两处）：

**1. `ai/train/args.py`** —— 在 `_get_ppo_config` 中添加：

```python
group.add_argument("--ent-coef", type=float, default=0.01,
                   help="PPO 熵正则系数，0=无正则（default: 0.01）")
group.add_argument("--vf-coef", type=float, default=0.5,
                   help="价值函数损失权重（default: 0.5）")
```

**2. `ai/algos/policy.py`** —— 在 `MaskablePPO(...)` 构造中添加：

```python
self._model = MaskablePPO(
    "MlpPolicy",
    env,
    ...
    ent_coef=args.ent_coef,     # 新增
    vf_coef=args.vf_coef,       # 新增
)
```

训练脚本中加 `--ent-coef 0.03 --vf-coef 0.25` 即可启用。

### 中优先级：价值函数正则化

| 参数 | 当前值 | 建议值 | 理由 |
|------|--------|--------|------|
| `vf_coef`（价值函数损失权重） | 默认 0.5 | **0.25 ~ 0.3** | 降低 Critic 在总 loss 中的权重，减缓 Critic 过拟合 |
| `--net-arch` | `256 256` | 可考虑 `256 128` 或 `128 128` | 减小价值函数网络的容量，降低过拟合风险 |

### 中优先级：改善探索

| 参数 | 当前值 | 建议值 | 理由 |
|------|--------|--------|------|
| `--pool-sampling-strategy` | `progress` | **`uniform`** | 均匀采样让 Agent 更频繁地遭遇各种强度的对手，避免针对特定对手过拟合 |
| `--sampling-lam` | 1.0 | **2.0 ~ 3.0** | 提高 temperature 让采样更均匀（仅在 progress/elo 策略下有效） |

### 低优先级：监控与预警

| 措施 | 理由 |
|------|------|
| 添加 `value_loss` 的 W&B alert | 当 value_loss 连续 N 个 checkpoint 上升时自动告警 |
| 添加 `entropy_loss > -1.5` 的 W&B alert | entropy 过高（即策略过于确定性）是崩溃的前兆 |
| 记录 pool acceptance rate | 如果 acceptance rate 持续走低，说明 ELO 门控可能正在制造死亡螺旋 |
| 每个 checkpoint 记录 `pool_size_by_region` | 确认池是否在正常增长 |

### 可选的架构级改动

| 方案 | 复杂度 | 效果 |
|------|--------|------|
| **Target Network for Critic**（类似 DQN） | 中 | 价值函数使用独立的、慢更新的 target 网络，减少 Critic 过拟合 |
| **Critic Ensemble**（多个价值函数取平均） | 高 | 减少单个 Critic 的偏差 |
| **Periodic Reset**（定期重置策略熵） | 低 | 每隔 N 步将 entropy_coef 临时提高，强制策略重新探索 |
| **GradNorm Clip on Critic** | 低 | 对价值函数的梯度做裁剪，防止 Critic 在单次更新中剧烈变化 |

## 五、验证方案

1. **最小验证**：添加 `--ent-coef 0.05`，用 region 4/20 重跑 500k steps，观察 entropy_loss 是否保持在 -2.0 以下
2. **对比验证**：A/B test：
   - A 组：`ent_coef=0.03, vf_coef=0.25, n_epochs=5`
   - B 组：当前默认参数
   - 对比 2M steps 后的 value_loss 和 ELO 轨迹
3. **长期验证**：用调参后的配置跑完整的 5M steps，确认不再出现崩溃

## 六、关键监控指标

训练中应持续关注的**早期预警信号**（按优先级排列）：

1. **value_loss** > 1000 且持续上升 → Critic 可能正在过拟合
2. **entropy_loss** > -1.5（即熵 < 1.5 nats） → 策略探索严重不足
3. **approx_kl** < 0.005 → 策略更新量过小，可能已停滞
4. **clip_fraction** < 0.03 → PPO clip 几乎不触发，梯度信号微弱
