import os
import unittest

from game.chat import ChatMessage, ChatRoom
from game.datatypes.game_map import Region
from game.datatypes.state import GameState
from llm.diplomat import LLMDiplomat
from llm.prompts import build_diplomat_system_prompt
from tests.helpers import map_with_regions


def _make_state(turn: int = 3) -> GameState:
    r1 = Region("北京", [], 5)
    r1.owner = 1
    r1.troops = 50
    r1.is_capital = True
    r2 = Region("上海", [], 3)
    r2.owner = 2
    r2.troops = 30
    r2.is_capital = True
    m = map_with_regions([None, r1, r2])
    return GameState(m, num_players=2, turn=turn)


def _chat_with_history() -> ChatRoom:
    room = ChatRoom()
    room.add_message(ChatMessage(sender_id=1, sender_name="玩家1", text="大家好，战争开始了。", turn=1))
    room.add_message(ChatMessage(sender_id=2, sender_name="玩家2", text="来吧，不要废话。", turn=1))
    room.add_message(ChatMessage(sender_id=1, sender_name="玩家1", text="玩家2，你的上海守不住了，趁早投降！", turn=2))
    return room


class _VerboseDiplomat(LLMDiplomat):
    """在测试中打印完整 prompt 和响应，供诊断。"""
    def _call(self, user_content: str) -> str:
        sep = "=" * 60
        print(f"\n{sep}\n[PROMPT SENT TO LLM]\n{user_content}\n{sep}")
        result = super()._call(user_content)
        print(f"[LLM RESPONSE] {result}\n{sep}")
        return result


@unittest.skipUnless(os.getenv("SILICONFLOW_API_KEY"), "SILICONFLOW_API_KEY not set")
class TestLLMDiplomatIntegration(unittest.TestCase):
    def setUp(self):
        system_prompt = build_diplomat_system_prompt("queen")
        self.diplomat = _VerboseDiplomat(system_prompt=system_prompt)

    def test_generate_empty_chat(self):
        """无聊天记录时能正常生成。"""
        msg = self.diplomat.generate_message(_make_state(), ChatRoom(), 2)
        self.assertIsInstance(msg, str)
        self.assertGreater(len(msg), 0)

    def test_generate_with_challenge(self):
        """最新一条是玩家1对玩家2的挑衅，玩家2应回应（人工看输出确认）。"""
        room = _chat_with_history()
        msg = self.diplomat.generate_message(_make_state(turn=3), room, 2)
        self.assertGreater(len(msg), 0)

    def test_history_order_in_prompt(self):
        """验证 get_history_text 返回逆序（最新在前），确保 LLM 优先读到最新消息。"""
        room = _chat_with_history()
        text = room.get_history_text(max_turns=5)
        lines = text.strip().splitlines()
        # 最新一条（回合2 玩家1 挑衅）应排在第一行
        self.assertIn("回合2", lines[0])
        self.assertIn("投降", lines[0])

    def test_generate_no_chat_response(self):
        """开局无任何历史，玩家2作为挑衅者主动发言。"""
        msg = self.diplomat.generate_message(_make_state(turn=1), ChatRoom(), 2)
        print(f"\n[开局发言] {msg}")
        self.assertGreater(len(msg), 0)

    def test_multi_turn_responds_to_player(self):
        """4 回合对话，验证 AI 不自说自话，能回应玩家的话（人工看输出）。
        每轮玩家说的话足够有辨识度，方便判断 AI 是否真的在接话。"""
        exchanges = [
            "你叫我弟弟？我才是大哥",
            "我拿下你两个省了，服不服",
            "我觉得你在靠运气",
            "你快输了，自己知道吗",
        ]
        room = ChatRoom()
        sep = "-" * 50
        print(f"\n{sep} 多轮回应测试 {sep}")
        for turn, player_text in enumerate(exchanges, start=1):
            # 玩家先说，AI 再回
            room.add_message(ChatMessage(sender_id=1, sender_name="玩家1", text=player_text, turn=turn))
            print(f"[回合{turn}] 玩家1: {player_text}")
            ai_msg = self.diplomat.generate_message(_make_state(turn=turn), room, 2)
            self.assertGreater(len(ai_msg), 0)
            room.add_message(ChatMessage(sender_id=2, sender_name="玩家2", text=ai_msg, turn=turn))
            print(f"[回合{turn}] 玩家2(AI): {ai_msg}")
        print(sep)
