# LwgEnv 环境设计文档

## 1. 概览

`LwgEnv`（`ai/envs/env.py`）是 LightWarGame 的 [Gymnasium](https://gymnasium.farama.org/) 封装，供强化学习 agent 训练使用。它将游戏引擎（`game/`）包装成标准的单智能体接口：agent 控制玩家 1，内置对手在每步自动行动。

```
env = LwgEnv("two_players/vsbaseline")
obs, info        = env.reset()
mask             = env.action_masks()          # invalid-action masking
obs, r, te, tr, info = env.step(action)
```

---

## 2. 观测空间（Observation Space）

### 2.1 接口

```python
Box(low=0.0, high=0.0, shape=(dim,), dtype=float32)
dim = num_regions * F
F   = max_players + 6
```

以 `max_players=6`、31 个地区为例：`F = 12`，`dim = 31 × 12 = 372`。

### 2.2 每个地区的编码（F 维）

| 偏移 | 字段 | 取值 | 说明 |
|------|------|------|------|
| `0` | `owner_neutral` | `{0, 1}` | owner == 中立 |
| `1` | `owner_self` | `{0, 1}` | owner == viewer 自身 |
| `2..max_players` | `owner_other_k` | `{0, 1}` | owner == 第 k 个其他玩家（player_id 升序） |
| `max_players+1` | `troops_norm` | `[0, 1]` | `troops / 500`，超出截断 |
| `max_players+2` | `is_capital` | `{0, 1}` | 是否为首都 |
| `max_players+3` | `base_growth_norm` | `[0, 1]` | `base_growth / 10` |
| `max_players+4` | `is_visible` | `{0, 1}` | 己方地区为 1，迷雾地区为 0 |
| `max_players+5` | `is_adj_to_mine` | `{0, 1}` | 与任意己方地区相邻（且不属于自己） |

**owner 使用 viewer-relative one-hot**：编码不包含绝对 player_id，对手永远映射到 `index 2+`，便于自对弈时角色互换。

### 2.3 战争迷雾

`Observation` 来自 `GameState.get_observation(player_id)`，非己方地区仅暴露 `owner`，`troops`/`is_capital`/`base_growth` 均为 `None`。编码时这些字段保持 0，并将 `is_visible` 置 0 区分。

---

## 3. 动作空间（Action Space）

### 3.1 接口

```python
Discrete(dim)
dim = len(edge_list) * 4 + 1
```

以中国地图（31 地区）为例：共 ~100 条有向边，动作数约 401。

### 3.2 动作编码

| 索引 | 含义 |
|------|------|
| `0` | **no-op**，本回合不出兵 |
| `1 + edge_idx * 4 + bucket_idx` | 沿边 `edge_idx` 出兵，比例档 `bucket_idx` |

**比例档**（`_TROOP_BUCKETS`）：

| `bucket_idx` | 出兵比例 | 实际兵力 |
|------|------|------|
| 0 | 25% | `max(1, floor(available × 0.25))` |
| 1 | 50% | `max(1, floor(available × 0.50))` |
| 2 | 75% | `max(1, floor(available × 0.75))` |
| 3 | 100% | `max(1, floor(available × 1.00))` |

`available = src.troops - 1`（始终留 1 守地）。

**边列表**（`_edges`）在 `ActionEncoder.__init__` 中从地图邻接关系静态构建，按 `(src_id, tgt_id)` 升序固定，之后不变。

### 3.3 动作掩码

`env.action_masks()` 返回长度 `dim` 的 `bool` 数组，供 invalid-action masking 使用（如 MaskablePPO）。

**`index 0`（no-op）始终为 True。**

`index k > 0` 合法的充要条件（三者同时满足）：

1. **配额未耗尽**：`commands_issued < max_commands`
   - `max_commands = max(1, owned_regions // 3)`
   - 拥有地区数越多，每回合可发出指令越多
2. **源地区归 agent 所有**：`r_obs.owner == viewer_id`
3. **源地区有兵可动**：`r_obs.troops > 1`

当条件 1 不满足时，直接返回只有 no-op 为 True 的掩码（整条边的 4 个动作均屏蔽）。

### 3.4 解码流程

```python
edge_idx, bucket_idx = divmod(action - 1, 4)
src_id, tgt_id = edge_list[edge_idx]
available = game_map.regions[src_id].troops - 1
troops = max(1, floor(available * bucket_ratio))
return Command(source=src_id, target=tgt_id, troops=troops, player=agent_id)
```

解码时从 **当前帧的真实地图**（而非观测快照）读取兵力，避免迷雾误差。

---

## 4. 奖励函数

奖励为多个函数的加权叠加，由 YAML 配置驱动。

| 函数 | 触发时机 | 说明 |
|------|------|------|
| `WinLoseReward` | 终局 | 胜 `+win`，败 `+lose`，超时平局 `0` |
| `TerritoryReward` | 每步 | `(己方地区变化 × territory_gain) + (敌方地区变化 × territory_loss)` |
| `CapitalCaptureReward` | 每步 | 每占领一个敌方首都 `+capital_capture` |

默认配置（`vsbaseline.yaml`）：

```
win=100, lose=-100, territory_gain=5, territory_loss=-5, capital_capture=20
```

奖励函数在 `reset()` 时调用 `rf.reset()`，支持有状态的 shaping（如累计地区数）。

---

## 5. 内置对手

对手在 `step()` 内部自动行动，agent 无需感知其存在。

| 类型 | 描述 |
|------|------|
| `RandomOpponent` | 从合法动作中均匀随机采样 |
| `RuleOpponent` | 基于规则：攻击配额优先打邻敌，调兵配额集中兵力到前线；攻击优先于调兵 |

配额计算与 agent 相同：`max(1, owned // 3)`，攻击占 `ceil(quota/2)`，调兵占 `floor(quota/2)`。

---

## 6. Episode 终止条件

| 条件 | 标志 |
|------|------|
| 游戏分出胜负（`GameState.settle()` 返回 True） | `terminated=True` |
| 达到最大回合数（`max_turns`，默认 150） | `truncated=True` |

---

## 7. 配置文件

配置位于 `ai/envs/configs/<name>.yaml`，通过 `LwgEnv(config_name)` 加载。

```yaml
game:
  map_config: cn          # data/map_configs/ 下的地图名
  num_players: 2
  max_players: 6          # 决定观测向量维度，与训练模型绑定
  max_turns: 150
  capital_mode: fixed     # fixed | random
  capitals: [8, 25]       # capital_mode=fixed 时的首都列表

reward:
  win: 100.0
  lose: -100.0
  shaped:
    territory_gain: 5.0
    territory_loss: -5.0
    capital_capture: 20.0

training:
  mode: vsbaseline        # vsbaseline | self_play
  opponent: random        # random | rule
```

> `max_players` 决定观测向量的维度，训练完成后不可更改（更改需重新训练）。
