"""Integration tests for web routes — Flask test client, real WebGameUi, no real game threads."""
import os
import shutil
import unittest
from pathlib import Path

from game.campaign.init_game import SESSIONS_DIR
from web import create_app
from web.web_game_ui import WebGameUi

_RUN = os.getenv("RUN_INTEGRATION")


def _cleanup(name):
    """Remove session directory, ignore errors from active Runner threads."""
    d = SESSIONS_DIR / name
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


@unittest.skipUnless(_RUN, "set RUN_INTEGRATION=1 to run")
class TestWebRoutes(unittest.TestCase):

    def setUp(self):
        _cleanup("_int_test")
        self.ui = WebGameUi()
        self.app = create_app(self.ui)

    def tearDown(self):
        _cleanup("_int_test")

    def test_index_returns_200(self):
        with self.app.test_client() as c:
            r = c.get("/")
            self.assertEqual(r.status_code, 200)
            self.assertIn(b"CREATE TEMPLATE", r.data)

    def test_create_session_writes_config(self):
        with self.app.test_client() as c:
            c.post("/create-session", data={
                "session_name": "_int_test",
                "num_players": "3",
                "capitals": "10,20,30",
            })
        cfg = SESSIONS_DIR / "_int_test" / "config.yaml"
        self.assertTrue(cfg.exists())

    def test_create_session_no_name_returns_400(self):
        with self.app.test_client() as c:
            r = c.post("/create-session", data={"session_name": "", "num_players": "2"})
            self.assertEqual(r.status_code, 400)

    def test_start_without_session_returns_400(self):
        with self.app.test_client() as c:
            r = c.post("/start", data={"session": ""})
            self.assertEqual(r.status_code, 400)

    def test_start_nonexistent_session_returns_404(self):
        with self.app.test_client() as c:
            r = c.post("/start", data={"session": "_nonexistent"})
            self.assertEqual(r.status_code, 404)

    def test_play_redirects_when_no_game(self):
        with self.app.test_client() as c:
            r = c.get("/play")
            self.assertEqual(r.status_code, 302)

    def test_wait_returns_200_when_lobby(self):
        with self.app.test_client() as c:
            r = c.get("/wait")
            self.assertEqual(r.status_code, 200)
            self.assertIn(b"AWAITING", r.data)

    def test_result_redirects_when_lobby(self):
        with self.app.test_client() as c:
            r = c.get("/result")
            self.assertEqual(r.status_code, 302)

    def test_command_done_without_game(self):
        with self.app.test_client() as c:
            r = c.post("/command", data={"done": "1"}, follow_redirects=True)
            self.assertEqual(r.status_code, 200)

    def test_double_start_blocked(self):
        with self.app.test_client() as c:
            c.post("/create-session", data={
                "session_name": "_int_test", "num_players": "2", "capitals": "1,2",
            })
            r1 = c.post("/start", data={"session": "_int_test", "force_new": "1"},
                       follow_redirects=True)
            self.assertEqual(r1.status_code, 200)
            # Second start should be blocked
            r2 = c.post("/start", data={"session": "_int_test", "force_new": "1"})
            self.assertIn(r2.status_code, (409, 302),
                          f"Expected 409 or 302, got {r2.status_code}")

    def test_error_phase_shows_traceback(self):
        self.ui.set_error("Something went wrong\n  File 'foo.py', line 42")
        with self.app.test_client() as c:
            r = c.get("/wait")
            self.assertEqual(r.status_code, 200)
            self.assertIn(b"Something went wrong", r.data)
