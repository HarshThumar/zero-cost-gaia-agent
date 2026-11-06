"""Publish the code to a free *static* HF Space (code mirror + landing page). Outward-facing:
run only with the user's OK.

HF requires a PRO plan to host Gradio Spaces on free CPU, so the Space is static: it hosts the
source (browsable at /tree/main, which the leaderboard links to) and index.html. No secrets are
uploaded — nothing on a static Space can use them. Run the agent locally (run_local.py / app.py).

  uv run python scripts/push_space.py            # dry run: lists files
  uv run python scripts/push_space.py --yes      # create space if needed, upload code
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
SPACE_NAME = "zero-cost-gaia-agent"
IGNORE = [".env", "CLAUDE.md", ".venv/*", "runs/*", "data/files/*", "data/cache/*", ".claude/*", ".git/*",
          "**/__pycache__/*", "*.pyc", "uv.lock"]
SPACE_HEADER = """---
title: Zero-Cost GAIA Agent
emoji: 🕵️
colorFrom: indigo
colorTo: blue
sdk: static
app_file: index.html
pinned: false
short_description: LangGraph GAIA agent on free Cloudflare Workers AI models
---

"""


def _files() -> list[str]:
    out = []
    for p in ROOT.rglob("*"):
        rel = p.relative_to(ROOT).as_posix()
        if p.is_file() and not any(Path(rel).match(g) or rel.startswith(g.rstrip("*")) for g in IGNORE):
            out.append(rel)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()

    api = HfApi(token=os.environ["HF_TOKEN"])
    repo_id = f"{api.whoami()['name']}/{SPACE_NAME}"
    files = _files()
    print(f"Space: https://huggingface.co/spaces/{repo_id}  (public, static, no secrets)")
    print(f"Files ({len(files)}): {files}")
    if not a.yes:
        print("Dry run. Re-run with --yes to push.")
        return

    api.create_repo(repo_id, repo_type="space", space_sdk="static", private=False, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=str(ROOT),
                      ignore_patterns=IGNORE + ["README.md"], commit_message="Update GAIA agent")
    # Spaces need YAML config at the top of README; GitHub's README stays clean without it.
    readme = SPACE_HEADER + (ROOT / "README.md").read_text(encoding="utf-8")
    api.upload_file(path_or_fileobj=readme.encode("utf-8"), path_in_repo="README.md",
                    repo_id=repo_id, repo_type="space", commit_message="Update README")
    print(f"Pushed → https://huggingface.co/spaces/{repo_id}/tree/main")


if __name__ == "__main__":
    main()
