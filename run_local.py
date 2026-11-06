"""Run the agent on the course questions locally and cache answers + traces.

  uv run python run_local.py                    # all 20
  uv run python run_local.py --ids 8e86,a1e9    # subset (task_id prefixes)
  uv run python run_local.py -j 4               # parallelism (default 3)

Writes runs/<timestamp>/{answers.json, traces/<task_id>.md} and merges into data/cache/answers.json
(the latest answer per task, which submit.py sends).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from agent.files import API_URL, get_attachment
from agent.graph import run, trace_markdown

ROOT = Path(__file__).parent
QFILE = ROOT / "data" / "questions.json"
CACHE = ROOT / "data" / "cache" / "answers.json"


def load_questions(refresh: bool) -> list[dict]:
    if refresh or not QFILE.exists():
        QFILE.write_text(json.dumps(requests.get(f"{API_URL}/questions", timeout=60).json(), indent=1), encoding="utf-8")
    return json.loads(QFILE.read_text(encoding="utf-8"))


def solve(q: dict, trace_dir: Path) -> dict:
    t0 = time.time()
    try:
        path = get_attachment(q["task_id"], q.get("file_name"))
        answer, msgs = run(q["question"], str(path) if path else None)
        trace = trace_markdown(msgs)
    except Exception as e:  # noqa: BLE001
        answer, trace = "", f"ERROR\n```\n{traceback.format_exc()}\n```"
        print(f"  ! {q['task_id'][:8]} failed: {e}")
    secs = round(time.time() - t0)
    (trace_dir / f"{q['task_id']}.md").write_text(
        f"# {q['task_id']}\n\n**Q:** {q['question']}\n\n**File:** {q.get('file_name') or '-'}\n\n"
        f"**Answer:** `{answer}`  ({secs}s)\n\n---\n\n{trace}", encoding="utf-8")
    return {"task_id": q["task_id"], "submitted_answer": answer, "seconds": secs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="", help="comma-separated task_id prefixes")
    ap.add_argument("-j", "--jobs", type=int, default=3)
    ap.add_argument("--refresh", action="store_true", help="re-download questions")
    a = ap.parse_args()

    qs = load_questions(a.refresh)
    if a.ids:
        pre = [p.strip() for p in a.ids.split(",") if p.strip()]
        qs = [q for q in qs if any(q["task_id"].startswith(p) for p in pre)]
    run_dir = ROOT / "runs" / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    (run_dir / "traces").mkdir(parents=True)
    print(f"Running {len(qs)} question(s) → {run_dir}")

    results = []
    with ThreadPoolExecutor(a.jobs) as ex:
        futs = {ex.submit(solve, q, run_dir / "traces"): q for q in qs}
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            print(f"  {r['task_id'][:8]}  {r['seconds']:>4}s  {r['submitted_answer']!r}")

    (run_dir / "answers.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    merged = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    for r in results:
        if r["submitted_answer"]:
            merged[r["task_id"]] = r["submitted_answer"]
    CACHE.write_text(json.dumps(merged, indent=1), encoding="utf-8")
    print(f"Cached answers: {len(merged)}/20 in {CACHE}")


if __name__ == "__main__":
    main()
