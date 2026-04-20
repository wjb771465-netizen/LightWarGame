# LightWarGame 测试规范

## 测试框架与入口

```bash
# 必须在仓库根目录下运行
cd /home/wjb/workspace/LightWarGame && conda run -n chinese_war_game python -m unittest discover -s tests -p "test_*.py" -v
```

`-v`：显示每条用例名，首个失败即可停止阅读，不需要在末尾找 FAILED 汇总。

**执行规则**：测试命令只跑一次，不裁剪输出。所有点、无 FAIL 即为通过。

---

## 改动后必须写测试并执行通过的范围

| 改动的模块 | 对应测试文件 |
|---|---|
| `game/datatypes/game_map.py` | `tests/test_game_map.py` |
| `game/datatypes/state.py` | `tests/test_game_state.py` |
| `game/datatypes/game_obs.py` | `tests/test_observation.py` |
| `game/runner.py` | `tests/test_runner.py` |
| `game/ui/display.py` | `tests/test_ui.py` |
| `game/ui/input_handler.py` | `tests/test_ui.py` |
| `game/ui/terminal_ui.py` | `tests/test_ui.py` |
| `game/save_load.py` | `tests/test_save_load.py` |

纯注释、类型注解、日志改动不强制要求测试。

---

## 测试结构

用 `unittest.TestCase` 组织，一个被测组件对应一个测试类，已有类中补用例，不新建散落文件。

**文件布局**：

```
tests/
├── helpers.py            # 共用工具（map_with_regions 等），不含测试用例
├── constants.py          # 共用常量（MAP_CONFIG）
├── test_game_map.py      # GameMap / Region
├── test_game_state.py    # GameState（settle / check_cmds / apply_cmds）
├── test_observation.py   # build_observation / Observation
├── test_runner.py        # GameRunner
├── test_ui.py            # display / input_handler / TerminalGameUi
└── test_save_load.py     # save_game / load_game
```

---

## 两类必须覆盖的测试

每次改动**至少覆盖以下两类**：

**1. 契约测试**：验证接口返回值的类型、形状与规格一致。

```python
# 例：settle 返回值类型
ended = s.settle()
self.assertIsInstance(ended, bool)
```

**2. 行为测试**：针对本次改动的逻辑，用最小场景断言具体结果。

```python
# 例：围困折半到达
a.owner = 1; a.troops = 10  # a 孤立（无友邻）
s.apply_cmds([Command(source=1, target=2, troops=4, player=1)])
# 实际到达 floor(4 * 0.5) = 2
self.assertEqual(b.troops, ...)
```

---

## 工具与约定

**构造最小地图**：用 `tests.helpers.map_with_regions`，不重复定义。

```python
from tests.helpers import map_with_regions as _map_with_regions

a = Region("a", [2], 4); a.owner = 1; a.troops = 10
b = Region("b", [1], 4); b.owner = 2; b.troops = 5
m = _map_with_regions([None, a, b])
```

**需要完整地图时**：用 `GameMap(MAP_CONFIG)`（从 `tests.constants` 引入 `MAP_CONFIG`）。

**UI 层测试**：注入 `io.StringIO` 和 `input_fn`，不真实读写终端。

```python
buf = io.StringIO()
ui = TerminalGameUi(out=buf, input_fn=lambda _: "y")
```

**文件系统测试**（save/load）：用 `tempfile.TemporaryDirectory`，测后自动清理，不留 `save.json` 在根目录。

```python
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "save.json")
    save_game(state, path)
    loaded = load_game(path)
```

**测试命名**：`test_<场景描述>_<期望结果>`，清楚表达意图，不用 `test_1` / `test_case_a`。

**不依赖随机性**：所有断言基于手动构造的确定场景，不调用无参 `GameMap()` 后直接断言 region 状态。

---

## 可以跳过的情况

- 纯重构（签名/行为不变），且现有测试已经过被改路径 → 说明原因后跳过
- 注释 / 日志 / 类型注解

跳过时须在回复中明确写出理由，不得静默略过。
