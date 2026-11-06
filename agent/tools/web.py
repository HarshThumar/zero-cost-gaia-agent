"""Web search (Serper), page fetching, and Wikipedia (incl. historical revisions)."""

from __future__ import annotations

import io
import os
import re
from functools import lru_cache

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from markdownify import markdownify

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36"}
WIKI_UA = {"User-Agent": "gaia-course-agent/0.1 (educational project)"}
CHUNK = 12000


def _paginate(text: str, offset: int, source: str) -> str:
    total = len(text)
    part = text[offset: offset + CHUNK]
    more = f"\n\n[... {total - offset - CHUNK} more chars; call again with offset={offset + CHUNK}]" if offset + CHUNK < total else ""
    return f"[{source} | chars {offset}-{min(offset + CHUNK, total)} of {total}]\n{part}{more}"


def _html_to_md(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "nav", "footer", "header", "noscript", "svg", "form"]):
        t.decompose()
    for t in soup.select(".mw-editsection, .reference, .reflist, .navbox, .metadata, .sistersitebox, #toc"):
        t.decompose()
    md = markdownify(str(soup), heading_style="ATX", strip=["img"])
    md = re.sub(r"\[([^\]]*)\]\((?:/wiki/|#)[^)]*\)", r"\1", md)  # drop internal wiki link targets
    return re.sub(r"\n{3,}", "\n\n", md).strip()


# Integrity: GAIA answers leak into public datasets, Spaces and repos. Never read them —
# the agent must find answers from primary sources.
_BLOCKED = re.compile(
    r"huggingface\.co/(datasets|spaces)/|hf\.co/(datasets|spaces)/|gaia-benchmark|"
    r"github\.com/[^ ]*gaia|gaia[-_ ]?(benchmark|validation|answers|solutions)",
    re.IGNORECASE,
)


def _blocked(*texts: str) -> bool:
    return any(_BLOCKED.search(t or "") for t in texts)


@lru_cache(maxsize=256)
def _fetch_text(url: str) -> str:
    if _blocked(url):
        raise PermissionError("blocked: benchmark-answer source (find the answer from primary sources)")
    r = requests.get(url, headers=UA, timeout=45)
    r.raise_for_status()
    ctype = r.headers.get("content-type", "")
    if "pdf" in ctype or url.lower().endswith(".pdf"):
        from pypdf import PdfReader

        return "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(r.content)).pages)
    if "html" in ctype or "<html" in r.text[:2000].lower():
        return _html_to_md(r.text)
    return r.text


@tool
def web_search(query: str) -> str:
    """Google search via Serper. Returns the top results (title, url, snippet) plus any answer box.
    Use specific queries; then open promising results with fetch_page."""
    r = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": os.environ["SERPER_API_KEY"]},
        json={"q": query, "num": 10},
        timeout=30,
    )
    r.raise_for_status()
    d = r.json()
    out = []
    if ab := d.get("answerBox"):
        out.append(f"ANSWER BOX: {ab.get('answer') or ab.get('snippet') or ab}")
    if kg := d.get("knowledgeGraph"):
        out.append(f"KNOWLEDGE GRAPH: {kg.get('title')} — {kg.get('description', '')} {kg.get('attributes', '')}")
    organic = [o for o in d.get("organic", []) if not _blocked(o.get("link", ""), o.get("title", ""), o.get("snippet", ""))]
    for i, o in enumerate(organic[:10], 1):
        out.append(f"{i}. {o.get('title')}\n   {o.get('link')}\n   {o.get('snippet', '')}")
    return "\n".join(out) or "No results."


@tool
def fetch_page(url: str, offset: int = 0) -> str:
    """Fetch a web page (or PDF) and return it as markdown text, 12k chars at a time.
    Use offset to read further. For Wikipedia prefer wikipedia_page."""
    try:
        return _paginate(_fetch_text(url), offset, url)
    except Exception as e:  # noqa: BLE001
        return f"Error fetching {url}: {e}"


@tool
def find_in_page(url: str, keywords: str) -> str:
    """Fetch a page and return only the passages containing any of the comma-separated keywords
    (with surrounding context). Good for long pages."""
    try:
        text = _fetch_text(url)
    except Exception as e:  # noqa: BLE001
        return f"Error fetching {url}: {e}"
    kws = [k.strip().lower() for k in keywords.split(",") if k.strip()]
    lines = text.splitlines()
    hits = []
    for i, ln in enumerate(lines):
        if any(k in ln.lower() for k in kws):
            hits.append("\n".join(lines[max(0, i - 3): i + 4]))
    return ("\n---\n".join(hits)[:CHUNK]) or f"No matches for {kws}."


def _wiki_api(params: dict) -> dict:
    params = {"format": "json", "formatversion": 2, **params}
    r = requests.get("https://en.wikipedia.org/w/api.php", params=params, headers=WIKI_UA, timeout=45)
    r.raise_for_status()
    return r.json()


@tool
def wikipedia_search(query: str) -> str:
    """Search English Wikipedia; returns matching article titles with snippets."""
    d = _wiki_api({"action": "query", "list": "search", "srsearch": query, "srlimit": 8})
    res = d.get("query", {}).get("search", [])
    return "\n".join(f"- {s['title']}: {re.sub('<[^>]+>', '', s['snippet'])}" for s in res) or "No results."


@lru_cache(maxsize=128)
def _wiki_text(title: str, as_of: str) -> str:
    params = {"action": "parse", "prop": "text", "redirects": 1}
    if as_of:
        rev = _wiki_api({
            "action": "query", "prop": "revisions", "titles": title, "rvlimit": 1,
            "rvstart": f"{as_of}T23:59:59Z", "rvdir": "older", "rvprop": "ids|timestamp", "redirects": 1,
        })
        pages = rev.get("query", {}).get("pages", [])
        if not pages or "revisions" not in pages[0]:
            raise ValueError(f"No revision of {title!r} on or before {as_of}")
        params["oldid"] = pages[0]["revisions"][0]["revid"]
    else:
        params["page"] = title
    d = _wiki_api(params)
    if "error" in d:
        raise ValueError(d["error"].get("info"))
    return _html_to_md(d["parse"]["text"])


@tool
def wikipedia_page(title: str, as_of_date: str = "", offset: int = 0) -> str:
    """Read an English Wikipedia article as markdown (tables included), 12k chars at a time.
    as_of_date (YYYY-MM-DD, optional) returns the latest revision on or before that date —
    use it when the question says e.g. 'use the 2022 version of Wikipedia' (then use 2022-12-31)."""
    try:
        return _paginate(_wiki_text(title, as_of_date), offset, f"wikipedia:{title}{'@' + as_of_date if as_of_date else ''}")
    except Exception as e:  # noqa: BLE001
        return f"Error: {e}"
