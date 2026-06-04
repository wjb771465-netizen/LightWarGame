"""game/constants.py 的单元测试：验证 max_commands() ceil 公式。"""

import unittest

from game.constants import max_commands, CMD_PER_TERRITORIES, MIN_COMMANDS


class TestMaxCommands(unittest.TestCase):
    """max_commands() ceil 公式验收标准。"""

    def test_zero_owned_returns_zero(self):
        """无领地时返回 0，即使有 MIN_COMMANDS 保底（调用方应先判断玩家存活）。"""
        self.assertEqual(max_commands(0), 0)

    def test_one_to_three_owned_returns_one(self):
        """1-3 地：ceil(owned/3) = 1，首升之前保持 1 条。"""
        for owned in (1, 2, 3):
            with self.subTest(owned=owned):
                self.assertEqual(max_commands(owned), 1)

    def test_first_upgrade_at_four(self):
        """4 地首升 → 2 条（ceil(4/3) = 2），旧 floor 公式需 6 地才升级。"""
        self.assertEqual(max_commands(4), 2)

    def test_five_to_six_owned_returns_two(self):
        """4-6 地保持 2 条。"""
        for owned in (5, 6):
            with self.subTest(owned=owned):
                self.assertEqual(max_commands(owned), 2)

    def test_second_upgrade_at_seven(self):
        """7 地 → 3 条（ceil(7/3) = 3）。"""
        self.assertEqual(max_commands(7), 3)

    def test_nine_owned_returns_three(self):
        """9 地仍是 3 条（ceil(9/3) = 3）。"""
        self.assertEqual(max_commands(9), 3)

    def test_third_upgrade_at_ten(self):
        """10 地 → 4 条。"""
        self.assertEqual(max_commands(10), 4)

    def test_twelve_owned_returns_four(self):
        self.assertEqual(max_commands(12), 4)

    def test_negative_returns_zero(self):
        """防御性：负数领地 → 0。"""
        self.assertEqual(max_commands(-1), 0)
        self.assertEqual(max_commands(-5), 0)

    def test_full_map_31_regions(self):
        """中国 31 省全占：ceil(31/3) = 11 条。"""
        self.assertEqual(max_commands(31), 11)

    def test_monotonic_non_decreasing(self):
        """配额随领地数单调不减。"""
        prev = 0
        for owned in range(1, 32):
            cur = max_commands(owned)
            self.assertGreaterEqual(cur, prev,
                                    f"not monotonic: {owned-1}→{prev}, {owned}→{cur}")
            prev = cur

    def test_respects_min_commands_bound(self):
        """任何正数领地至少获得 MIN_COMMANDS 条。"""
        for owned in range(1, 32):
            with self.subTest(owned=owned):
                self.assertGreaterEqual(max_commands(owned), MIN_COMMANDS)


if __name__ == "__main__":
    unittest.main()
