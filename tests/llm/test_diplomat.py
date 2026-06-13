import unittest
from unittest.mock import MagicMock, patch

from game.chat import ChatRoom
from llm.diplomat import LLMDiplomat
from llm.prompts import build_diplomat_system_prompt
from tests.helpers import map_with_regions
from game.datatypes.state import GameState
from game.datatypes.game_map import Region


def _make_state() -> GameState:
    r1 = Region("北京", [], 5)
    r1.owner = 1
    r1.troops = 50
    r1.is_capital = True
    r2 = Region("上海", [], 3)
    r2.owner = 2
    r2.troops = 30
    r3 = Region("天津", [], 2)
    r3.owner = 0
    r3.troops = 10
    m = map_with_regions([None, r1, r2, r3])
    state = GameState(m, num_players=2, turn=5)
    return state


def _make_diplomat(response: str = "局势有趣") -> LLMDiplomat:
    with patch("llm.base.OpenAI"):
        d = LLMDiplomat(system_prompt="test")
    d._call = MagicMock(return_value=response)
    return d


class TestPromptBuilder(unittest.TestCase):
    def test_default_persona_in_output(self):
        result = build_diplomat_system_prompt("default")
        self.assertIn("沉着", result)

    def test_aggressive_persona_in_output(self):
        result = build_diplomat_system_prompt("aggressive")
        self.assertIn("军阀", result)

    def test_template_slot_filled(self):
        result = build_diplomat_system_prompt("default")
        self.assertNotIn("{persona}", result)

    def test_unknown_persona_raises(self):
        with self.assertRaises(FileNotFoundError):
            build_diplomat_system_prompt("nonexistent_xyz")


class TestLLMDiplomat(unittest.TestCase):
    def test_generate_message_returns_api_response(self):
        d = _make_diplomat("你已无路可退")
        state = _make_state()
        result = d.generate_message(state, ChatRoom(), 1)
        self.assertEqual(result, "你已无路可退")

    def test_generate_message_calls_call_once(self):
        d = _make_diplomat()
        d.generate_message(_make_state(), ChatRoom(), 1)
        d._call.assert_called_once()

    def test_user_prompt_contains_turn(self):
        d = _make_diplomat()
        d.generate_message(_make_state(), ChatRoom(), 1)
        user_prompt = d._call.call_args[0][0]
        self.assertIn("回合", user_prompt)

    def test_user_prompt_contains_chat_history(self):
        d = _make_diplomat()
        room = ChatRoom()
        from game.chat import ChatMessage
        room.add_message(ChatMessage(2, "玩家2", "你好弱", 4))
        d.generate_message(_make_state(), room, 1)
        user_prompt = d._call.call_args[0][0]
        self.assertIn("你好弱", user_prompt)

    def test_render_state_classifies_regions(self):
        d = _make_diplomat()
        state = _make_state()
        text = d._render_state(state, 1)
        self.assertIn("北京", text)
        self.assertIn("上海", text)
