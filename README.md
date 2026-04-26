# LightWarGame

回合制战棋游戏。目前纯终端运行， 地图包括中国内地行政区（不含港澳台）。

## 环境配置

```bash
conda env create -f environment.yml
```

## 运行

```bash
conda run -n chinese_war_game python main.py
```

启动后：

1. 选择新游戏或读取存档
2. 新游戏：输入人数（2–6）→ 选随机首都或手动指定首都
3. 选择是否启用 AI 玩家及哪些玩家由 AI 控制

### AI 对战

编辑 `ai_players.json` 设置训练好的模型路径，启动时选择对应玩家为 AI 即可：

```json
{ "2": "saves/model.zip" }
```

AI 玩家回合自动推进，人类玩家回合正常输入指令。`obs_dim` 从模型权重自动推断，无需手动配置。

## 玩法

每回合各玩家依次输入指令，格式：

```
源地区编号,目标地区编号,兵力
```

空行或达到本回合上限（`max(1, 领地数 // 3)`）后结束输入，随后统一结算。

**核心规则：**

- 只能从己方地区向相邻地区派兵，出发地至少留 1 兵
- 出发地被敌方完全包围（孤立）时，途中兵力折半（向下取整）
- 多方同时进入同一地区：最强者胜，剩余兵力 = 最强值 − 次强值
- 并列最强时守方胜；守方不在并列中则该地变中立
- 占领所有对手首都后游戏结束
- 每回合结束后按地区增长率自动补充兵力

## 存档

每回合结算后自动保存至 `saves/save.json`，启动时选 `[2]` 可读取。

## 地图

`data/map_configs/cn.json`，31 个省级行政区，含邻接关系与基础增长率。

## 测试

```bash
cd /home/wjb/workspace/LightWarGame
conda run -n chinese_war_game python -m unittest discover -s tests -p "test_*.py" -v
```

## 项目结构

```
game/
├── datatypes/       # Region, GameMap, GameState, Command, Observation
├── ui/              # 终端显示、指令输入、地图渲染（含 AIGameUi）
├── init_game.py     # 开局初始化（随机首都 / 指定首都 / 读档）
├── runner.py        # 主循环
└── save_load.py     # 存读档

ai/
├── algos/           # Policy 接口与 SB3Policy 包装
├── envs/            # Gymnasium 环境封装、编解码、奖励函数、对手
├── renders/         # 地图渲染与视频生成
└── train/           # 训练脚本与配置

main.py              # 终端入口（GameLauncher）
ai_players.json      # AI 玩家模型路径配置
data/                # 地图数据
tests/               # 单元测试
```
