"""Read local attachments: spreadsheets, PDFs, docs, code, images, audio."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import tool

from agent import llm

IMAGE_EXT = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}


def _read(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".xlsx", ".xls", ".xlsm"):
        import pandas as pd

        sheets = pd.read_excel(path, sheet_name=None)
        return "\n\n".join(f"## Sheet: {name} ({len(df)} rows)\n{df.to_csv(index=False)}" for name, df in sheets.items())
    if ext == ".csv":
        return path.read_text(encoding="utf-8", errors="replace")
    if ext == ".pdf":
        from pypdf import PdfReader

        return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    if ext == ".docx":
        import docx

        d = docx.Document(str(path))
        parts = [p.text for p in d.paragraphs]
        for t in d.tables:
            parts += [" | ".join(c.text for c in row.cells) for row in t.rows]
        return "\n".join(parts)
    if ext == ".json":
        return json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=1)
    if ext in IMAGE_EXT:
        return "This is an image. Use describe_image (or chess_image_to_fen for chess boards)."
    if ext in AUDIO_EXT:
        return "This is audio. Use transcribe_audio."
    return path.read_text(encoding="utf-8", errors="replace")


@tool
def read_file(path: str) -> str:
    """Read a local file and return its text content. Handles xlsx/csv (as CSV per sheet),
    pdf, docx, json, py/txt/md. For images use describe_image; for audio use transcribe_audio.
    For calculations over spreadsheet data, prefer run_python with pandas on the same path."""
    p = Path(path)
    if not p.exists():
        return f"File not found: {path}"
    text = _read(p)
    return text if len(text) < 40000 else text[:40000] + f"\n[truncated; {len(text)} chars total — use run_python]"


@tool
def describe_image(path: str, question: str) -> str:
    """Ask a vision model a question about a local image file. Be specific in the question
    (e.g. 'list every piece and its square', 'transcribe all text')."""
    p = Path(path)
    mime = IMAGE_EXT.get(p.suffix.lower(), "image/png")
    return llm.vision(question, p.read_bytes(), mime)


@tool
def transcribe_audio(path: str) -> str:
    """Transcribe a local audio file (mp3/wav/m4a...) to text with Whisper."""
    return llm.transcribe(Path(path).read_bytes()) or "(empty transcript)"
