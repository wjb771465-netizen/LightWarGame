from __future__ import annotations

import threading
from pathlib import Path

from flask import current_app, redirect, render_template, request, send_file, url_for

from game.campaign.init_game import SESSIONS_DIR, from_session, list_sessions
from game.constants import max_commands
from game.datatypes.command import Command
from game.runner import GameRunner
from game.utils import get_saves_dir


def register_routes(app):
    web_ui = app.config["web_ui"]

    @app.route("/")
    def index():
        sessions = []
        for s in list_sessions():
            import json
            import yaml
            save_dir = get_saves_dir(s.name)
            save_file = save_dir / "save.json"
            has_save = save_file.exists()
            turn = None
            if has_save:
                try:
                    data = json.loads(save_file.read_text(encoding="utf-8"))
                    turn = data.get("turn")
                except Exception:
                    pass
            with open(s / "config.yaml", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            sessions.append({
                "name": s.name,
                "has_save": has_save,
                "turn": turn,
                "num_players": cfg.get("num_players", "?"),
                "has_ai": bool(cfg.get("ai_players")),
            })
        in_game = web_ui.snapshot()["phase"] not in ("lobby", "over")
        return render_template("index.html", sessions=sessions, in_game=in_game)

    @app.route("/create-session", methods=["POST"])
    def create_session():
        session_name = request.form.get("session_name", "").strip()
        if not session_name:
            return "Session name required", 400
        session_dir = SESSIONS_DIR / session_name
        session_dir.mkdir(parents=True, exist_ok=True)
        _create_session_config(session_dir, request.form)
        return redirect(url_for("index"))

    @app.route("/start", methods=["POST"])
    def start():
        session_name = request.form.get("session", "")
        force_new = request.form.get("force_new", "0") == "1"
        if not session_name:
            return "No session selected", 400

        snap = web_ui.snapshot()
        if snap["phase"] not in ("lobby", "over"):
            return "A game is already in progress. Refresh to rejoin.", 409

        session_dir = SESSIONS_DIR / session_name
        if not session_dir.is_dir():
            return f"Session '{session_name}' not found.", 404

        save_dir = get_saves_dir(session_name)
        is_new = force_new or not (save_dir / "save.json").exists()

        web_ui.load_ai_config(session_dir)
        web_ui.set_log_path(save_dir / "ai_decision.log")
        web_ui.prepare_launch(session_dir, is_new)

        threading.Thread(target=_run_game, args=(session_dir, is_new, web_ui),
                         daemon=True).start()

        return redirect(url_for("wait"))

    @app.route("/play")
    def play():
        snap = web_ui.snapshot()
        if snap["phase"] in ("result", "over"):
            return redirect(url_for("result"))
        if snap["phase"] == "lobby":
            return redirect(url_for("wait"))
        if snap["observation"] is None:
            return redirect(url_for("wait"))

        error = snap.get("cmd_error")
        if error:
            web_ui._cmd_error = None  # 消费一次即清

        # 计算每个己方地区的可用兵力配额
        quotas: dict[int, int] = {}
        for cmd in snap["pending_commands"]:
            src = cmd["source_name"]
            src_id = next(
                (r["id"] for r in snap["observation"]["regions"] if r["name"] == src), None
            )
            if src_id is not None:
                quotas[src_id] = quotas.get(src_id, 0) + cmd["troops"]

        region_quotas = {}
        for r in snap["observation"]["regions"]:
            if r["owner"] == snap["current_player"] and r["troops"] is not None:
                pending = quotas.get(r["id"], 0)
                region_quotas[r["id"]] = r["troops"] - 1 - pending

        # 计算当前玩家本回合指令上限
        owned_count = sum(
            1 for r in snap["observation"]["regions"]
            if r["owner"] == snap["current_player"]
        )
        cmd_limit = max_commands(owned_count)

        return render_template("play.html",
                               obs=snap["observation"],
                               pending=snap["pending_commands"],
                               turn=snap["turn"],
                               player_id=snap["current_player"],
                               quotas=region_quotas,
                               cmd_limit=cmd_limit,
                               error=error,
                               session_name=snap.get("session_name", ""),
                               chat_messages=snap.get("chat_messages", []),
                               map_path=snap.get("map_path", ""))

    @app.route("/command", methods=["POST"])
    def command():
        if "done" in request.form:
            web_ui.submit_commands()
            return redirect(url_for("wait"))

        source = int(request.form["source"])
        target = int(request.form["target"])
        troops = int(request.form["troops"])
        snap = web_ui.snapshot()
        cmd = Command(source=source, target=target, troops=troops,
                      player=snap["current_player"])
        regions = snap["observation"]["regions"] if snap["observation"] else []
        error = web_ui.add_pending_command(cmd, regions)
        if error:
            owned_count = sum(
                1 for r in snap["observation"]["regions"]
                if r["owner"] == snap["current_player"]
            ) if snap["observation"] else 0
            # 计算 region quotas（与正常路径一致）
            quotas: dict[int, int] = {}
            for cmd in snap["pending_commands"]:
                src = cmd["source_name"]
                src_id = next(
                    (r["id"] for r in snap["observation"]["regions"] if r["name"] == src), None
                ) if snap["observation"] else None
                if src_id is not None:
                    quotas[src_id] = quotas.get(src_id, 0) + cmd["troops"]
            region_quotas = {}
            if snap["observation"]:
                for r in snap["observation"]["regions"]:
                    if r["owner"] == snap["current_player"] and r["troops"] is not None:
                        pending = quotas.get(r["id"], 0)
                        region_quotas[r["id"]] = r["troops"] - 1 - pending
            return render_template("play.html",
                                   obs=snap["observation"],
                                   pending=snap["pending_commands"],
                                   turn=snap["turn"],
                                   player_id=snap["current_player"],
                                   quotas=region_quotas,
                                   cmd_limit=max_commands(owned_count),
                                   error=error,
                                   session_name=snap.get("session_name", ""),
                                   chat_messages=snap.get("chat_messages", []),
                                   map_path=snap.get("map_path", ""))
        return redirect(url_for("play"))

    @app.route("/wait")
    def wait():
        snap = web_ui.snapshot()
        if snap["phase"] == "error":
            return render_template("error.html", error=snap.get("error", "Unknown error"))
        if snap["phase"] == "playing" and snap["observation"] is not None:
            return redirect(url_for("play"))
        if snap["phase"] in ("result", "over"):
            return redirect(url_for("result"))
        return render_template("wait.html", turn=snap["turn"],
                               info=snap.get("cmd_error"),
                               chat_messages=snap.get("chat_messages", []))

    @app.route("/result")
    def result():
        snap = web_ui.snapshot()
        if snap["phase"] == "playing":
            return redirect(url_for("play"))
        if snap["phase"] in ("lobby", "waiting"):
            return redirect(url_for("wait"))
        is_over = snap["phase"] == "over"
        return render_template("turn_result.html",
                               turn=snap["turn"],
                               changes=snap["battle_changes"],
                               game_over=is_over,
                               winner=snap["winner"])

    @app.route("/map-image/<session_name>")
    def map_image(session_name):
        snap = web_ui.snapshot()
        map_path = snap.get("map_path")
        if map_path:
            return send_file(map_path, mimetype="image/png")
        # 回退：尝试找最新回合地图
        from game.utils import get_saves_dir
        maps_dir = get_saves_dir(session_name) / "maps"
        if maps_dir.is_dir():
            pngs = sorted(maps_dir.glob("map_turn_*.png"))
            if pngs:
                return send_file(str(pngs[-1]), mimetype="image/png")
        return "", 404

    @app.route("/chat", methods=["POST"])
    def chat():
        snap = web_ui.snapshot()
        text = request.form.get("text", "").strip()
        if text and snap["current_player"]:
            web_ui.add_chat_message(snap["current_player"], text)
        return redirect(url_for("play"))


def _create_session_config(session_dir: Path, form) -> None:
    import yaml
    num_players = int(form.get("num_players", "2"))
    capitals_raw = form.get("capitals", "").strip()
    cfg = {"name": session_dir.name, "num_players": num_players}
    if capitals_raw:
        cfg["capitals"] = [int(x.strip()) for x in capitals_raw.split(",")]
    else:
        cfg["capitals"] = "random"
    with open(session_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)


def _run_game(session_dir: Path, is_new: bool, web_ui) -> None:
    import logging
    import traceback
    import matplotlib
    matplotlib.use("Agg")

    from game.campaign.chat import ChatRoom

    try:
        save_dir = get_saves_dir(session_dir.name)
        save_dir.mkdir(parents=True, exist_ok=True)

        chat_room = ChatRoom()
        chat_path = save_dir / "chat.json"
        if not is_new and chat_path.exists():
            chat_room.load(str(chat_path))

        state = from_session(
            session_dir,
            save_path=None if is_new else save_dir / "save.json",
        )
        GameRunner(
            state, web_ui,
            save_path=save_dir,
            chat_room=chat_room if web_ui.has_ai_players else None,
        ).run()
    except Exception:
        web_ui.set_error(traceback.format_exc())
        logging.exception("Game runner crashed")
