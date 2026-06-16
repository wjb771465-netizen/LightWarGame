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
| `ai/algos/` | Policy 接口与 SB3Policy 包装（MaskablePPO） |
| `ai/envs/` | Gymnasium 环境封装（LwgEnv）、观测/动作编解码、奖励函数、内置对手 |
| `ai/renders/` | 地图渲染与视频生成（帧 PNG → ffmpeg 合成为视频） |
| `ai/train/` | 训练脚本、配置（sb3_trainer、args、YAML 配置） |
| `main.py` | 终端入口（GameLauncher），处理启动交互与 AI 玩家绑定 |
| `data/map_configs/` | 地图配置（cn.json：省份、邻接关系、增长率） |
| `tests/` | 测试（game / ai / llm / integration），详见下方 Test Conventions |

## Rules
- AI 玩家配置在 session 目录下的 JSON（通过 `game/campaign/init_game.py` 的 `load_session_config()` 加载），key 为 player_id（从 1 开始），含模型路径、外交官开关、persona 等字段
- `obs_dim` 从模型权重自动推断 `max_players`，训练后不可更改（改 max_players 需重新训练）
- 新增运行时依赖必须同步更新 `environment.yml`（conda 依赖放顶层，pip 依赖放 `pip:` 下）

## Test Conventions

- **Framework**: `unittest`，命令 `conda run -n chinese_war_game python -m unittest tests.xxx -v`
- **目录**: `tests/game/` 测 game 层，`tests/ai/` 测 ai 层，`tests/integration/` 跨层端到端
- **小地图**: `tests/helpers.map_with_regions(regions)` — 绕过配置加载直接注入 `Region` 列表（index 0 为 None，1-indexed）。所有非端到端测试都用 2-3 个手造 Region，不依赖 `data/map_configs/cn.json`
- **Region 手造模式**: `r = Region("name", [adjacent_ids], base_growth)` → 直接设 `r.owner`、`r.troops`、`r.is_capital`
- **Mock Policy**: 实现 `predict(obs, mask) -> int` 的可编程对象（按预设顺序返回动作，记录 `last_obs`/`last_mask` 用于断言），不加载真实模型权重
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
