import os
import tempfile
import unittest

from game.campaign.chat import ChatMessage, ChatRoom


class TestChatRoom(unittest.TestCase):
    def setUp(self):
        self.room = ChatRoom()
        self.msg1 = ChatMessage(1, "玩家1", "你好", 1)
        self.msg2 = ChatMessage(2, "玩家2", "你坏", 2)

    def test_add_and_get_all(self):
        self.room.add_message(self.msg1)
        self.room.add_message(self.msg2)
        self.assertEqual(len(self.room.get_history()), 2)

    def test_get_history_since_turn(self):
        self.room.add_message(self.msg1)
        self.room.add_message(self.msg2)
        result = self.room.get_history(since_turn=2)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].turn, 2)

    def test_get_history_text_empty(self):
        self.assertIn("无外交记录", self.room.get_history_text())

    def test_get_history_text_format(self):
        self.room.add_message(self.msg1)
        text = self.room.get_history_text()
        self.assertIn("玩家1", text)
        self.assertIn("你好", text)
        self.assertIn("回合1", text)

    def test_save_and_load(self):
        self.room.add_message(self.msg1)
        self.room.add_message(self.msg2)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "chat.json")
            self.room.save(path)
            room2 = ChatRoom()
            room2.load(path)
            history = room2.get_history()
            self.assertEqual(len(history), 2)
            self.assertEqual(history[0].text, "你好")
            self.assertEqual(history[1].sender_id, 2)

    def test_load_nonexistent_is_noop(self):
        self.room.load("/nonexistent/path/chat.json")
        self.assertEqual(self.room.get_history(), [])
