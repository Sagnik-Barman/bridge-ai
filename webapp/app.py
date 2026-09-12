"""Local Flask web app for playing a long bridge session against the
bot: full guided bidding (suggestion + explanation shown before every
call), standard board/vulnerability rotation, IMP-flavored scoring,
undo (single call/card, or restart the whole board), and save/resume —
including mid-board, via `game/persistence.py`.

Run with:

    python webapp/app.py

then open http://127.0.0.1:5000 in a browser.

Single-user, single-process session — this is a local practice tool for
one person, not a multi-user server. The current game is autosaved to
saves/_autosave.json after every action and reloaded automatically on
startup, so restarting the process no longer loses your progress. Named
saves ("save this game as ...") let you keep multiple practice games and
come back to any one of them later.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, render_template, request

from engine.deal import parse_card
from game.session import GameSession, IllegalActionError
from game import persistence

app = Flask(__name__)

# Single global session — see module docstring for why this is fine here.
SESSION: GameSession = persistence.load_autosave() or GameSession(starting_board=1)


def _autosave() -> None:
    try:
        persistence.autosave(SESSION)
    except OSError:
        pass  # autosave is a convenience, never worth failing the request over


def state_response(extra: dict | None = None) -> "flask.Response":
    payload = SESSION.to_dict()
    if extra:
        payload.update(extra)
    return jsonify(payload)


def error_response(message: str, status: int = 400):
    return jsonify({"error": message}), status


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state", methods=["GET"])
def get_state():
    return state_response()


@app.route("/api/call", methods=["POST"])
def make_call():
    data = request.get_json(force=True) or {}
    call = data.get("call")
    if not call:
        return error_response("Missing 'call'.")
    try:
        feedback = SESSION.make_user_call(call)
    except IllegalActionError as exc:
        return error_response(str(exc))
    _autosave()
    return state_response({"feedback": feedback})


@app.route("/api/undo_call", methods=["POST"])
def undo_call():
    ok = SESSION.undo_last_call()
    _autosave()
    return state_response({"undo_ok": ok})


@app.route("/api/card", methods=["POST"])
def play_card():
    data = request.get_json(force=True) or {}
    seat = data.get("seat")
    card_str = data.get("card")
    if not seat or not card_str:
        return error_response("Missing 'seat' or 'card'.")
    try:
        card = parse_card(card_str)
    except ValueError as exc:
        return error_response(str(exc))
    try:
        SESSION.play_user_card(seat, card)
    except IllegalActionError as exc:
        return error_response(str(exc))
    _autosave()
    return state_response()


@app.route("/api/undo_card", methods=["POST"])
def undo_card():
    ok = SESSION.undo_last_card()
    _autosave()
    return state_response({"undo_ok": ok})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        analysis = SESSION.analyze_current_position()
    except IllegalActionError as exc:
        return error_response(str(exc))
    return jsonify({"analysis": analysis})


@app.route("/api/restart_board", methods=["POST"])
def restart_board():
    SESSION.restart_current_board()
    _autosave()
    return state_response()


@app.route("/api/next_board", methods=["POST"])
def next_board():
    if SESSION.phase != "board_complete":
        return error_response("Finish the current board before moving on.")
    SESSION.next_board()
    _autosave()
    return state_response()


@app.route("/api/new_session", methods=["POST"])
def new_session():
    global SESSION
    data = request.get_json(force=True) or {}
    starting_board = int(data.get("starting_board", 1))
    play_mode = data.get("play_mode", "heuristic")
    try:
        SESSION = GameSession(starting_board=starting_board, play_mode=play_mode)
    except ValueError as exc:
        return error_response(str(exc))
    _autosave()
    return state_response()


# -- save / resume -----------------------------------------------------

@app.route("/api/saves", methods=["GET"])
def list_saves():
    return jsonify({"saves": persistence.list_saves()})


@app.route("/api/save", methods=["POST"])
def save_game():
    data = request.get_json(force=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return error_response("Missing 'name' to save under.")
    if name == persistence.AUTOSAVE_NAME:
        return error_response("That name is reserved for autosave — pick another.")
    persistence.save_session(SESSION, name)
    return jsonify({"saved": True, "name": name, "saves": persistence.list_saves()})


@app.route("/api/load", methods=["POST"])
def load_game():
    global SESSION
    data = request.get_json(force=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return error_response("Missing 'name' to load.")
    try:
        SESSION = persistence.load_session(name)
    except FileNotFoundError:
        return error_response(f"No saved game named {name!r}.", status=404)
    except (ValueError, KeyError, OSError) as exc:
        return error_response(f"Couldn't load {name!r}: {exc}")
    _autosave()
    return state_response()


@app.route("/api/saves/<name>", methods=["DELETE"])
def delete_save(name: str):
    ok = persistence.delete_save(name)
    if not ok:
        return error_response(f"No saved game named {name!r}.", status=404)
    return jsonify({"deleted": True, "saves": persistence.list_saves()})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
