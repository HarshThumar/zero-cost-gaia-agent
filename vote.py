"""Self-consistency: majority-vote each task's answer across several local runs.

  uv run python vote.py --since 20261106-181159     # use runs at/after this timestamp

Writes data/cache/answers.json (what submit.py sends) and prints the per-task vote table.
Ties go to the most recent run. Empty answers never win.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent


def _elementwise(answers: list[str]) -> str | None:
    """No whole-answer majority: for same-length ordered lists, vote each position separately.
    e.g. ['A, B', 'A, C', 'D, B'] → 'A, B'. Returns None if lists differ in length or a slot has no majority."""
    parts = [[p.strip() for p in a.split(",")] for a in answers]
    if len({len(p) for p in parts}) != 1 or len(parts[0]) < 2:
        return None
    out = []
    for slot in zip(*parts):
        c = Counter(x.lower() for x in slot)
        word, n = c.most_common(1)[0]
        if n * 2 <= len(slot):
            return None
        out.append(next(x for x in reversed(slot) if x.lower() == word))
    return ", ".join(out)


def pick_answer(answers: list[str]) -> str:
    """Majority vote (case-insensitive); ties → most recent; no majority on a list → per-position vote."""
    answers = [a for a in answers if a]
    if not answers:
        return ""
    c = Counter(x.lower() for x in answers)
    top = max(c.values())
    winners = {k for k, v in c.items() if v == top}
    pick = next(x for x in reversed(answers) if x.lower() in winners)
    if top == 1 and len(answers) > 2:
        pick = _elementwise(answers) or pick
    return pick


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", required=True, help="run folder name (timestamp) to start from")
    a = ap.parse_args()

    runs = sorted(d for d in (ROOT / "runs").iterdir() if d.name >= a.since and (d / "answers.json").exists())
    votes: dict[str, list[str]] = {}
    for d in runs:  # oldest → newest
        for r in json.loads((d / "answers.json").read_text(encoding="utf-8")):
            if r["submitted_answer"]:
                votes.setdefault(r["task_id"], []).append(r["submitted_answer"])

    final = {}
    print(f"{len(runs)} runs: {[d.name for d in runs]}\n")
    for tid, answers in sorted(votes.items()):
        # Count case-insensitively, but keep the most recent original spelling.
        pick = pick_answer(answers)
        c = Counter(x.lower() for x in answers)
        top = max(c.values())
        final[tid] = pick
        agree = "" if len(c) == 1 else "  ⚠ split"
        print(f"{tid[:8]}  {top}/{len(answers)}  {pick!r}{agree}   {dict(c) if len(c) > 1 else ''}")

    out = ROOT / "data" / "cache" / "answers.json"
    out.write_text(json.dumps(final, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(final)} voted answers → {out}")


if __name__ == "__main__":
    main()
