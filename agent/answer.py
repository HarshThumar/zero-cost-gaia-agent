"""Turn the model's final message into a bare, exact-match-friendly answer."""

from __future__ import annotations

import re

# A single number. Commas count as thousands separators only with a $ or a decimal part
# ("$89,706.00", "1,234.5"); otherwise "132,133,134" is a list, not 132133134.
_NUM = re.compile(r"^(\$\s*-?(\d{1,3}(,\d{3})+|\d+)(\.\d+)?|-?(\d{1,3}(,\d{3})+)\.\d+|-?\d+(\.\d+)?)\s*%?$")
_SPECIAL = re.compile(r"<\|[^|>]*\|>")  # gpt-oss channel tokens like <|return|>, <|end|>


def extract_final(text: str) -> str:
    """Take what follows the last 'FINAL ANSWER:' marker (or the last non-empty line)."""
    text = _SPECIAL.sub("", text or "").strip()
    m = list(re.finditer(r"final\s*answer\s*[:：]\s*", text, re.IGNORECASE))
    if m:
        text = text[m[-1].end():]
    else:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        text = lines[-1] if lines else ""
    return text.strip()


def normalize(ans: str) -> str:
    a = _SPECIAL.sub("", ans or "").strip()
    a = a.splitlines()[0].strip() if "\n" in a else a
    for _ in range(2):  # peel wrappers in any order: **x**. / "x". / `x`
        a = a.rstrip(".").strip()  # GAIA answers never end with a sentence period
        a = a.strip("*`").strip()
        if len(a) >= 2 and a[0] == a[-1] and a[0] in "\"'":
            a = a[1:-1].strip()
    if _NUM.match(a):  # plain number: drop $, thousands commas, %
        a = a.replace("$", "").replace(",", "").replace("%", "").strip()
    a = re.sub(r"\s*,\s*", ", ", a) if "," in a else a  # canonical list spacing
    return a


def final_answer(text: str) -> str:
    return normalize(extract_final(text))
