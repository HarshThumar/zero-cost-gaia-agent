"""YouTube: transcript (captions → Whisper fallback) and frame sampling → vision."""

from __future__ import annotations

import glob
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langchain_core.tools import tool

from agent import llm

CACHE = Path(__file__).resolve().parents[2] / "data" / "cache" / "video"


def _vid(url: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/|shorts/)([\w-]{11})", url)
    return m.group(1) if m else re.sub(r"\W", "_", url)[-40:]


def _ytdlp(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "yt_dlp", "--no-playlist", "--quiet", "--no-warnings", *args],
                          capture_output=True, text=True, timeout=600)


def _vtt_to_text(vtt: str) -> str:
    out, last = [], None
    for ln in vtt.splitlines():
        ln = re.sub(r"<[^>]+>", "", ln).strip()
        if not ln or "-->" in ln or ln.startswith(("WEBVTT", "Kind:", "Language:")) or ln.isdigit():
            continue
        if ln != last:
            out.append(ln)
            last = ln
    return " ".join(out)


@tool
def youtube_transcript(url: str) -> str:
    """Get what is SAID in a YouTube video: captions if available, else Whisper on the audio."""
    vid = _vid(url)
    d = CACHE / vid
    d.mkdir(parents=True, exist_ok=True)
    cached = d / "transcript.txt"
    if cached.exists():
        return cached.read_text(encoding="utf-8")
    _ytdlp("--skip-download", "--write-subs", "--write-auto-subs", "--sub-langs", "en.*,en", "--sub-format", "vtt",
           "-o", str(d / "subs"), url)
    vtts = glob.glob(str(d / "subs*.vtt"))
    if vtts:
        text = _vtt_to_text(Path(vtts[0]).read_text(encoding="utf-8", errors="replace"))
    else:
        r = _ytdlp("-f", "bestaudio[ext=m4a]/bestaudio", "-o", str(d / "audio.%(ext)s"), url)
        auds = glob.glob(str(d / "audio.*"))
        if not auds:
            return f"Could not get captions or audio: {r.stderr[-800:]}"
        mp3 = d / "audio16k.mp3"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", auds[0], "-ac", "1", "-ar", "16000", "-b:a", "48k", str(mp3)], check=True)
        text = llm.transcribe(mp3.read_bytes())
    cached.write_text(text, encoding="utf-8")
    return text or "(empty transcript)"


def _duration(path: str) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


@tool
def youtube_frames(url: str, frame_question: str, max_frames: int = 60) -> str:
    """Answer a question about what is SEEN in a YouTube video. Samples frames evenly across the WHOLE
    video and asks a vision model `frame_question` about each frame independently.
    Write frame_question about a single frame. If you need a number per frame (e.g. how many distinct
    bird species are visible at once), say so: each frame answer then ends with COUNT=<n>, and the tool
    reports the maximum COUNT and where it occurred. Vision models sometimes hallucinate — look at the
    per-frame details and prefer counts that several frames agree on."""
    if not shutil.which("ffmpeg"):
        return "ffmpeg not installed."
    vid = _vid(url)
    d = CACHE / vid
    d.mkdir(parents=True, exist_ok=True)
    vids = glob.glob(str(d / "video.*"))
    if not vids:
        r = _ytdlp("-f", "bv*[height<=720][ext=mp4]+ba/b[height<=720]/b", "--merge-output-format", "mp4", "-o", str(d / "video.%(ext)s"), url)
        vids = glob.glob(str(d / "video.*"))
        if not vids:
            return f"Video download failed: {r.stderr[-800:]}"
    dur = _duration(vids[0]) or 120.0
    max_frames = max(4, min(int(max_frames), 90))
    every = max(1.0, dur / max_frames)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", vids[0], "-vf", f"fps=1/{every:.3f},scale=768:-2",
                        "-frames:v", str(max_frames), f"{tmp}/f_%04d.jpg"], check=True)
        frames = sorted(Path(tmp).glob("f_*.jpg"))
        prompt = (f"{frame_question}\nLook only at THIS single video frame. Be careful and literal: do not guess "
                  "species or objects that are not clearly visible. Answer briefly. If the question asks for a "
                  "number, end your answer with COUNT=<n> (COUNT=0 if none).")

        def ask(i_f):
            i, f = i_f
            try:
                return i * every, llm.vision(prompt, f.read_bytes(), "image/jpeg").strip().replace("\n", " ")
            except Exception as e:  # noqa: BLE001
                return i * every, f"error {e}"

        with ThreadPoolExecutor(4) as ex:
            answers = list(ex.map(ask, enumerate(frames)))
    counts = [(int(m.group(1)), t) for t, a in answers if (m := re.search(r"COUNT\s*=\s*(\d+)", a))]
    summary = ""
    if counts:
        mx = max(c for c, _ in counts)
        at = [f"{t:.0f}s" for c, t in counts if c == mx]
        hist = {}
        for c, _ in counts:
            hist[c] = hist.get(c, 0) + 1
        summary = f"MAX COUNT={mx} (at {', '.join(at)}); count histogram {dict(sorted(hist.items()))}\n"
    lines = "\n".join(f"t={t:.0f}s: {a}" for t, a in answers)
    return f"Video {dur:.0f}s, {len(frames)} frames (every {every:.1f}s).\n{summary}{lines}"
