"""Submit cached answers (data/cache/answers.json) to the course leaderboard.

  uv run python submit.py            # dry run: shows what would be sent
  uv run python submit.py --yes      # actually submit

The username is read from HF_TOKEN (whoami). agent_code points at the public Space.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()
API_URL = "https://agents-course-unit4-scoring.hf.space"
SPACE = "zero-cost-gaia-agent"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()

    user = requests.get("https://huggingface.co/api/whoami-v2",
                        headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}, timeout=30).json()["name"]
    answers = json.loads(Path("data/cache/answers.json").read_text(encoding="utf-8"))
    payload = {
        "username": user,
        "agent_code": f"https://huggingface.co/spaces/{user}/{SPACE}/tree/main",
        "answers": [{"task_id": k, "submitted_answer": v} for k, v in answers.items()],
    }
    print(f"user={user}  answers={len(payload['answers'])}  agent_code={payload['agent_code']}")
    if not a.yes:
        print("Dry run. Re-run with --yes to submit.")
        return
    r = requests.post(f"{API_URL}/submit", json=payload, timeout=120)
    print(r.status_code, json.dumps(r.json(), indent=1))


if __name__ == "__main__":
    main()
