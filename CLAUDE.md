# LightWarGame

## Background
回合制战棋游戏，地图覆盖中国 31 个内地省级行政区（不含港澳台）。当前阶段：终端可玩 + 强化学习 AI 已集成（MaskablePPO via SB3），支持 AI 对战与人类玩家混排。

## Key Paths
| 路径 | 用途 |
|------|------|
| `game/datatypes/` | 核心数据类型：Region、GameMap、GameState、Command、Observation |
| `game/ui/` | 终端 UI（显示、地图渲染、指令输入、AI 自动出招） |
| `game/campaign/` | 战役包：开局初始化、存读档（save_load）、聊天室（chat） |
| `game/runner.py` | 游戏主循环：展示 → 收集指令 → 校验 → 结算 |
| `game/ui_ports.py` | UI 协议（GameUiPort），Runner 通过它解耦 UI 实现 |
| `llm/` | LLM 外交官层：diplomat（发言）、director、base、prompts/ |
| `ai/algos/` | Policy 接口与 SB3Policy 包装（MaskablePPO），opponent/region pool，采样策略，GNN backbone |
| `ai/envs/` | Gymnasium 环境封装（LwgEnv）、观测/动作编解码、奖励函数、内置对手 |
| `ai/envs/opponents/` | 内置对手：random、rule、FSM、policy |
| `ai/envs/rewards/` | 奖励函数：领土、首都、胜负、步数惩罚 |
| `ai/renders/` | 地图渲染与视频生成（帧 PNG → ffmpeg 合成为视频） |
| `ai/train/` | 训练入口（`__main__.py`）、配置（args）、训练器（sb3/self_play/region_self_play）、评估（eval）、指标（metrics）、路径工具（utils） |
| `ai/train/scripts/` | 训练启动脚本（Shell） |
| `tests/` | 测试（game / ai / llm / integration / smoke），详见下方 Test Conventions |
| `claude-plans/` | 设计文档与方案讨论 |
| `main.py` | 终端入口（GameLauncher），处理启动交互与 AI 玩家绑定 |
| `data/map_configs/` | 地图配置（cn.json：省份、邻接关系、增长率） |

## Rules
- AI 玩家配置在 session 目录下的 JSON（通过 `game/campaign/init_game.py` 的 `load_session_config()` 加载），key 为 player_id（从 1 开始），含模型路径、外交官开关、persona 等字段
- `obs_dim` 从模型权重自动推断 `max_players`，训练后不可更改（改 max_players 需重新训练）
- 新增运行时依赖必须同步更新 `environment.yml`（conda 依赖放顶层，pip 依赖放 `pip:` 下）
- Encoder config 通过 `self._model._config` 属性持久化（SB3 无 `custom_objects` API，直接挂属性利用 cloudpickle 序列化）

## Smoke Tests

`tests/smoke/train.py` 覆盖训练全流程：

```
conda run -n chinese_war_game python -m tests.smoke.train --scenario duel/vsbaseline
```

4 个 Phase：optimizer check → forward activation check → 全流程训练（save + eval + render）→ gradient/param/product 检查。产出 checkpoint zip、final model zip、tensorboard log、eval video。

## Test Conventions

- **Framework**: `unittest`，通过项目根目录的 `Makefile` 运行：
  - `make test` — 全量（跳过集成测试，需要 `RUN_INTEGRATION` 或 `SILICONFLOW_API_KEY` 环境变量）
  - `make test-ai` — 仅 `tests/ai/`
  - `make test-game` — 仅 `tests/game/`
  - `make test-integration` — 仅 `tests/integration/`
  - 单文件：`conda run -n chinese_war_game python -m unittest tests.xxx -v`
  - discover 必须带 `-t .` 保证 import 路径正确
- **目录**: `tests/game/` 测 game 层，`tests/ai/` 测 ai 层，`tests/integration/` 跨层端到端
- **小地图**: `tests/helpers.map_with_regions(regions)` — 绕过配置加载直接注入 `Region` 列表（index 0 为 None，1-indexed）。所有非端到端测试都用 2-3 个手造 Region，不依赖 `data/map_configs/cn.json`
- **Region 手造模式**: `r = Region("name", [adjacent_ids], base_growth)` → 直接设 `r.owner`、`r.troops`、`r.is_capital`
- **Mock Policy**: 实现 `predict(obs, mask) -> int` 的可编程对象（按预设顺序返回动作，记录 `last_obs`/`last_mask` 用于断言），不加载真实模型权重
- **纯 NN 模块测试**: 只测张量进出（forward shape、eval 确定性），不依赖 SB3、GameMap、YAML。`GNNBackbone` 等纯 `nn.Module` 的测试归入 `tests/ai/test_nn.py`
- **Test class > method**: `TestXxx` 继承 `unittest.TestCase`，方法 `test_xxx` 带中文或英文 docstring
- **setUp/setUpClass**: 共享 fixture 放 `setUp()` 中构造；简单的 factory method 也行（`_linear_map()` 等返回 state/map）
- **跳过集成测试**: `@unittest.skipUnless(os.getenv("RUN_INTEGRATION"), "...")`，默认不跑昂贵测试
- **临时文件**: 用 `tempfile.TemporaryDirectory`，不污染仓库
- **无 pytest/conftest/fixture 装饰器**: 保持纯 unittest 风格
- **回归要求**: 新增模块对应新增 test class；改现有代码必须跑全量已有测试

## Tech Stack
- Python 3.11，conda 环境（chinese_war_game）
- Gymnasium + Stable-Baselines3 + sb3-contrib（MaskablePPO）
- PyTorch >= 2.0
- NumPy, Matplotlib（渲染）
- wandb, tensorboard（训练监控）

## W&B 诊断

参见 `.claude/skills/wandb-inspect/skill.md`。
