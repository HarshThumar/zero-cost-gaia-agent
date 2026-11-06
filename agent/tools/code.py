"""Python execution sandbox (subprocess, timeout) and chess tools."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from langchain_core.tools import tool

from agent import llm


@tool
def run_python(code: str) -> str:
    """Run a Python script in a fresh subprocess and return stdout/stderr (60s timeout).
    pandas, openpyxl, chess, requests are available. Always print() the values you need.
    Use it for arithmetic, counting, date math, data analysis, string manipulation, logic checks."""
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "snippet.py"
        f.write_text(code, encoding="utf-8")
        try:
            r = subprocess.run([sys.executable, str(f)], capture_output=True, text=True, timeout=60, cwd=d)
        except subprocess.TimeoutExpired:
            return "Error: timed out after 60s."
    out = (r.stdout or "") + (f"\n[stderr]\n{r.stderr}" if r.stderr else "")
    return out[:15000] or "(no output — did you print?)"


def _stockfish_path() -> str | None:
    cands = [os.getenv("STOCKFISH_PATH"), shutil.which("stockfish"), "/usr/games/stockfish", "/usr/bin/stockfish"]
    pkgs = Path(os.getenv("LOCALAPPDATA", "")) / "Microsoft/WinGet/Packages"
    if pkgs.exists():
        cands += [str(p) for p in pkgs.glob("Stockfish*/**/stockfish*.exe")]
    return next((c for c in cands if c and Path(c).exists()), None)


ORIENT_PROMPT = """This is a chess board diagram with coordinate labels printed inside the edge squares.
Read the rank number printed in the TOP-LEFT square and the file letter printed in the BOTTOM-LEFT square.
Reply exactly as: TOPLEFT=<number> BOTTOMLEFT=<letter>"""

SQUARE_PROMPT = """This image is ONE square cropped from a chess board diagram (ignore any small
coordinate letter/number in the corner). Is there a chess piece on it?
Reply with exactly one of: empty, white king, white queen, white rook, white bishop, white knight,
white pawn, black king, black queen, black rook, black bishop, black knight, black pawn"""

_PIECE = {"king": "k", "queen": "q", "rook": "r", "bishop": "b", "knight": "n", "pawn": "p"}


def _classify(img_bytes: bytes) -> str:
    """Return FEN letter for one square ('.' for empty), or '?' if unreadable."""
    ans = llm.vision(SQUARE_PROMPT, img_bytes, "image/png").lower()
    if "empty" in ans and not any(k in ans for k in _PIECE):
        return "."
    for name, letter in _PIECE.items():
        if name in ans:
            return letter.upper() if "white" in ans else letter
    return "?"


@tool
def chess_image_to_fen(path: str, side_to_move: str = "w") -> str:
    """Read a chess board IMAGE into a FEN (side_to_move: 'w' or 'b'). Handles boards shown from
    either side by reading the printed coordinates. Crops the board into 64 squares and identifies
    each one, so it is far more reliable than describe_image for chess. If the board is valid, the
    output already includes Stockfish's best move — use it rather than writing your own search."""
    import io
    from concurrent.futures import ThreadPoolExecutor

    import chess
    from PIL import Image

    raw = Path(path).read_bytes()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = img.size
    side = min(w, h)
    img = img.crop(((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2))
    cell = side / 8

    orient = llm.vision(ORIENT_PROMPT, raw, "image/png").upper()
    flipped = "TOPLEFT=1" in orient.replace(" ", "") or "BOTTOMLEFT=H" in orient.replace(" ", "")

    def crop(rc):
        r, c = rc
        sq = img.crop((round(c * cell), round(r * cell), round((c + 1) * cell), round((r + 1) * cell))).resize((192, 192))
        buf = io.BytesIO()
        sq.save(buf, "PNG")
        return _classify(buf.getvalue())

    cells = [(r, c) for r in range(8) for c in range(8)]
    with ThreadPoolExecutor(8) as ex:
        labels = list(ex.map(crop, cells))

    board_grid = {}  # (rank 1-8, file 0-7) -> letter
    for (r, c), lab in zip(cells, labels):
        rank, file = (r + 1, 7 - c) if flipped else (8 - r, c)
        board_grid[(rank, file)] = lab

    fen_rows, unknown = [], 0
    for rank in range(8, 0, -1):
        s, empty = "", 0
        for f in range(8):
            x = board_grid[(rank, f)]
            if x == "?":
                unknown += 1
                x = "."
            if x == ".":
                empty += 1
            else:
                s += (str(empty) if empty else "") + x
                empty = 0
        fen_rows.append(s + (str(empty) if empty else ""))
    fen = "/".join(fen_rows) + f" {side_to_move} - - 0 1"
    try:
        b = chess.Board(fen)
        status = "valid" if b.is_valid() else f"INVALID ({b.status()!r})"
        diagram = str(b)
    except ValueError as e:
        status, diagram = f"unparseable: {e}", ""
    orientation = "black at bottom (flipped)" if flipped else "white at bottom"
    engine = ""
    if status == "valid":  # run the engine here so the agent doesn't try to hand-roll a search
        engine = "\nStockfish: " + chess_best_move.invoke({"fen": fen, "seconds": 5.0})
    return (f"FEN: {fen}\nOrientation: {orientation} [{orient.strip()}]\n"
            f"Board status: {status}; unreadable squares: {unknown}\n{diagram}{engine}")


@tool
def chess_best_move(fen: str, seconds: float = 3.0) -> str:
    """Return Stockfish's best move for a FEN position in SAN and UCI, plus the evaluation and main line."""
    import chess
    import chess.engine

    sf = _stockfish_path()
    if not sf:
        return "Stockfish not installed."
    board = chess.Board(fen)
    with chess.engine.SimpleEngine.popen_uci(sf) as eng:
        info = eng.analyse(board, chess.engine.Limit(time=seconds))
    pv = info.get("pv", [])
    if not pv:
        return "No move found."
    san_line = board.variation_san(pv[:6])
    return f"Best move: {board.san(pv[0])} (uci {pv[0].uci()}), score {info.get('score')}, line: {san_line}"
