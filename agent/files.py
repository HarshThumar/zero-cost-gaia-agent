"""Fetch question attachments.

Order: local cache → course API /files/{task_id} → GAIA dataset (attachment file ONLY).
The course endpoint currently 404s for every attachment, hence the fallback.
We download exactly `2023/validation/<file_name>` — never metadata.jsonl (it holds the answers).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://agents-course-unit4-scoring.hf.space"
FILES_DIR = Path(__file__).resolve().parent.parent / "data" / "files"


def get_attachment(task_id: str, file_name: str | None) -> Path | None:
    if not file_name:
        return None
    if "/" in file_name or "\\" in file_name or "metadata" in file_name.lower():
        raise ValueError(f"Refusing suspicious attachment name: {file_name!r}")
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    dest = FILES_DIR / file_name
    if dest.exists() and dest.stat().st_size > 0 and not dest.read_bytes()[:12].startswith(b'{"detail"'):
        return dest

    try:
        r = requests.get(f"{API_URL}/files/{task_id}", timeout=60)
        if r.ok and r.content and not r.content.startswith(b'{"detail"'):
            dest.write_bytes(r.content)
            return dest
    except requests.RequestException:
        pass

    from huggingface_hub import hf_hub_download

    p = hf_hub_download(
        repo_id="gaia-benchmark/GAIA",
        repo_type="dataset",
        filename=f"2023/validation/{file_name}",
        token=os.getenv("HF_TOKEN"),
    )
    shutil.copy(p, dest)
    return dest
