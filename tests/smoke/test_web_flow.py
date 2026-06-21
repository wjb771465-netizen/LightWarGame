"""Smoke tests — real Runner thread, real game turns. Requires RUN_INTEGRATION=1."""
import os
import shutil
import time
import unittest
from pathlib import Path

from game.campaign.init_game import SESSIONS_DIR
from web import create_app
from web.web_game_ui import WebGameUi

_RUN = os.getenv("RUN_INTEGRATION")


def _cleanup(name):
    d = SESSIONS_DIR / name
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


def _wait_for_phase(ui, phases, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = ui.snapshot()
        if s["phase"] in phases:
            return s
        time.sleep(0.1)
    return ui.snapshot()


def _wait_for_player(ui, player_id, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = ui.snapshot()
        if s["phase"] == "error":
            raise RuntimeError(f"Runner crashed: {s.get('error')}")
        if s["observation"] is not None and s["current_player"] == player_id:
            return s
        time.sleep(0.1)
    return ui.snapshot()


@unittest.skipUnless(_RUN, "set RUN_INTEGRATION=1 to run")
class TestWebFullGame(unittest.TestCase):
    """端到端：人类 vs 人类完整对局。"""

    def setUp(self):
        _cleanup("_smoke_full")
        self.ui = WebGameUi()
        self.app = create_app(self.ui)

    def tearDown(self):
        _cleanup("_smoke_full")

    def test_two_human_players_one_turn(self):
        """两人各下一轮 → 战报 → turn 推进。"""
        with self.app.test_client() as c:
            c.post("/create-session", data={
                "session_name": "_smoke_full", "num_players": "2", "capitals": "20,4",
            })
            c.post("/start", data={"session": "_smoke_full", "force_new": "1"})

            # P1
            s = _wait_for_player(self.ui, 1)
            self.assertEqual(s["turn"], 1)
            c.post("/command", data={"source": "20", "target": "21", "troops": "10"})
            c.post("/command", data={"done": "1"})

            # P2
            s = _wait_for_player(self.ui, 2)
            self.assertEqual(s["turn"], 1)
            c.post("/command", data={"source": "4", "target": "5", "troops": "10"})
            c.post("/command", data={"done": "1"})

            # Battle result
            s = _wait_for_phase(self.ui, ("result", "playing"))
            self.assertIn(s["phase"], ("result", "playing"))

            # Should progress to turn 2
            s = _wait_for_player(self.ui, 1, timeout=15)
            self.assertEqual(s["turn"], 2, f"Turn stuck at {s['turn']}, error={s.get('error')}")


@unittest.skipUnless(_RUN, "set RUN_INTEGRATION=1 to run")
class TestWebAIFlow(unittest.TestCase):
    """端到端：AI 对手 + 外交官配置。"""

    def setUp(self):
        _cleanup("_smoke_ai")
        self.ui = WebGameUi()
        self.app = create_app(self.ui)

    def tearDown(self):
        _cleanup("_smoke_ai")

    def test_session_with_ai_random_opponent(self):
        """含 AI random 配置 → AI 自动出招 → 人类正常操作。"""
        session_dir = SESSIONS_DIR / "_smoke_ai"
        session_dir.mkdir(parents=True)
        import yaml
        with open(session_dir / "config.yaml", "w") as f:
            yaml.dump({
                "name": "_smoke_ai",
                "num_players": 2,
                "capitals": [20, 4],
                "ai_players": {
                    2: {"type": "random"}
                },
            }, f)

        with self.app.test_client() as c:
            c.post("/start", data={"session": "_smoke_ai", "force_new": "1"})

            # AI (P2) should be skipped automatically → human P1 shows up
            s = _wait_for_player(self.ui, 1, timeout=15)
            self.assertIsNotNone(s["observation"],
                                 f"P1 obs is None — error={s.get('error')}")
            self.assertEqual(s["current_player"], 1)

            # Human submits
            c.post("/command", data={"source": "20", "target": "21", "troops": "10"})
            c.post("/command", data={"done": "1"})

            # Should resolve (AI P2 already auto-responded)
            s = _wait_for_phase(self.ui, ("result", "playing"), timeout=15)
            self.assertIn(s["phase"], ("result", "playing"))

    def test_session_with_diplomat_does_not_block(self):
        """外交官配置不应导致 Runner 崩溃或阻塞。"""
        session_dir = SESSIONS_DIR / "_smoke_ai"
        session_dir.mkdir(parents=True)
        import yaml
        with open(session_dir / "config.yaml", "w") as f:
            yaml.dump({
                "name": "_smoke_ai",
                "num_players": 2,
                "capitals": [20, 4],
                "ai_players": {
                    2: {"type": "random", "diplomat": True}
                },
            }, f)

        with self.app.test_client() as c:
            c.post("/start", data={"session": "_smoke_ai", "force_new": "1"})

            s = _wait_for_player(self.ui, 1, timeout=15)
            self.assertNotEqual(s["phase"], "error",
                                f"Runner crashed on diplomat session: {s.get('error')}")
            self.assertEqual(s["current_player"], 1)

            c.post("/command", data={"source": "20", "target": "21", "troops": "10"})
            c.post("/command", data={"done": "1"})

            s = _wait_for_phase(self.ui, ("result", "playing"), timeout=15)
            self.assertNotEqual(s["phase"], "error",
                                f"Runner crashed after commands: {s.get('error')}")


@unittest.skipUnless(_RUN, "set RUN_INTEGRATION=1 to run")
class TestWebLoadContinue(unittest.TestCase):
    """端到端：读档继续。"""

    def setUp(self):
        _cleanup("_smoke_load")
        self.ui = WebGameUi()
        self.app = create_app(self.ui)

    def tearDown(self):
        _cleanup("_smoke_load")

    def test_save_and_continue(self):
        """打一回合存盘 → 新 WebGameUi 读档 → 再打一回合。"""
        with self.app.test_client() as c:
            # Step 1: new game, play through 1 turn
            c.post("/create-session", data={
                "session_name": "_smoke_load", "num_players": "2", "capitals": "20,4",
            })
            c.post("/start", data={"session": "_smoke_load", "force_new": "1"})

            s = _wait_for_player(self.ui, 1)
            c.post("/command", data={"source": "20", "target": "21", "troops": "10"})
            c.post("/command", data={"done": "1"})

            s = _wait_for_player(self.ui, 2)
            c.post("/command", data={"source": "4", "target": "3", "troops": "10"})
            c.post("/command", data={"done": "1"})

            _wait_for_phase(self.ui, ("result",))
            s = _wait_for_player(self.ui, 1, timeout=15)
            self.assertEqual(s["turn"], 2, f"Turn stuck at {s['turn']}")

        # Step 2: new WebGameUi, load save
        ui2 = WebGameUi()
        app2 = create_app(ui2)
        with app2.test_client() as c:
            # Not force_new → loads save.json
            c.post("/start", data={"session": "_smoke_load"})

            s = _wait_for_player(ui2, 1, timeout=15)
            self.assertNotEqual(s["phase"], "error",
                                f"Load crashed: {s.get('error')}")
            self.assertEqual(s["turn"], 2, f"Loaded turn is {s['turn']}, expected 2")

            c.post("/command", data={"source": "20", "target": "21", "troops": "5"})
            c.post("/command", data={"done": "1"})

            s = _wait_for_player(ui2, 2, timeout=15)
            c.post("/command", data={"source": "4", "target": "5", "troops": "5"})
            c.post("/command", data={"done": "1"})

            _wait_for_phase(ui2, ("result",))
            s = _wait_for_player(ui2, 1, timeout=15)
            self.assertEqual(s["turn"], 3,
                             f"After continue turn stuck at {s['turn']}")
