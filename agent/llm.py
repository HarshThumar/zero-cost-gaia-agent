"""Cloudflare Workers AI clients with automatic account rotation.

Accounts come from CF_AI_ACCOUNT_ID_n / CF_AI_TOKEN_n (n = 1..N). When a call fails with a
rate-limit / daily-quota error, we move to the next account and retry. The current account
index is sticky (shared across threads) so we don't keep hammering an exhausted account.
"""

from __future__ import annotations

import base64
import os
import threading
import time

import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

AGENT_MODEL = os.getenv("AGENT_MODEL", "@cf/openai/gpt-oss-120b")
VISION_MODEL = os.getenv("VISION_MODEL", "@cf/google/gemma-4-26b-a4b-it")
ASR_MODEL = os.getenv("ASR_MODEL", "@cf/openai/whisper-large-v3-turbo")

_QUOTA_MARKERS = ("429", "rate limit", "too many requests", "quota", "allocation", "neurons", "capacity")


def _load_accounts() -> list[tuple[str, str]]:
    accounts = []
    for n in range(1, 50):
        acc, tok = os.getenv(f"CF_AI_ACCOUNT_ID_{n}"), os.getenv(f"CF_AI_TOKEN_{n}")
        if acc and tok:
            accounts.append((acc, tok))
    # Also accept un-numbered vars (handy for a single-account Space).
    acc, tok = os.getenv("CF_AI_ACCOUNT_ID"), os.getenv("CF_AI_TOKEN")
    if acc and tok and (acc, tok) not in accounts:
        accounts.append((acc, tok))
    if not accounts:
        raise RuntimeError("No Cloudflare accounts found (set CF_AI_ACCOUNT_ID_1 / CF_AI_TOKEN_1).")
    return accounts


ACCOUNTS = _load_accounts()
_lock = threading.Lock()
_current = 0
_clients: dict[tuple[int, str, float], ChatOpenAI] = {}


def is_quota_error(err: Exception | str) -> bool:
    s = str(err).lower()
    return any(m in s for m in _QUOTA_MARKERS)


def _advance(from_idx: int) -> None:
    global _current
    with _lock:
        if _current == from_idx:  # only advance once per exhausted account
            _current = (_current + 1) % len(ACCOUNTS)


def chat_model(model: str = AGENT_MODEL, idx: int | None = None, temperature: float = 0.0) -> ChatOpenAI:
    idx = _current if idx is None else idx
    key = (idx, model, temperature)
    if key not in _clients:
        acc, tok = ACCOUNTS[idx]
        _clients[key] = ChatOpenAI(
            model=model,
            api_key=tok,
            base_url=f"https://api.cloudflare.com/client/v4/accounts/{acc}/ai/v1",
            temperature=temperature,
            timeout=180,
            max_retries=0,  # we rotate/retry ourselves; client retries honor long Retry-After on 429
            max_tokens=8192,
        )
    return _clients[key]


def _cf_compatible(messages):
    """Cloudflare rejects assistant messages whose content is null (LangChain sends null when an
    AIMessage only carries tool calls). Give those a short placeholder string."""
    from langchain_core.messages import AIMessage

    fixed = []
    for m in messages:
        if isinstance(m, AIMessage) and (m.tool_calls or m.invalid_tool_calls) and not m.content:
            m = m.model_copy(update={"content": "(calling tools)"})
        fixed.append(m)
    return fixed


def invoke_rotating(messages, tools=None, model: str = AGENT_MODEL, temperature: float = 0.0):
    """Invoke a chat model, rotating accounts on quota errors and backing off on transient ones.
    Bad-request (400) errors are raised immediately — retrying can't fix them."""
    if isinstance(messages, list) and messages and not isinstance(messages[0], dict):
        messages = _cf_compatible(messages)
    last_err: Exception | None = None
    attempts = len(ACCOUNTS) + 3
    for attempt in range(attempts):
        idx = _current
        llm = chat_model(model, idx, temperature)
        if tools:
            llm = llm.bind_tools(tools)
        try:
            return llm.invoke(messages)
        except Exception as e:  # noqa: BLE001 — provider errors come in many shapes
            last_err = e
            status = getattr(e, "status_code", None)
            body = getattr(getattr(e, "response", None), "text", "")
            if is_quota_error(e) or is_quota_error(body) or status == 429:
                _advance(idx)
            elif status == 400 and "context" not in body.lower():
                if os.getenv("CF_DEBUG_DIR"):
                    import json, uuid
                    payload = chat_model(model, idx)._get_request_payload(messages, tools=[t for t in (tools or [])])
                    p = os.path.join(os.environ["CF_DEBUG_DIR"], f"bad-{uuid.uuid4().hex[:6]}.json")
                    with open(p, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=1, default=str)
                raise RuntimeError(f"Cloudflare 400: {body[:1000]}") from e
            else:
                time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"All Cloudflare attempts failed: {last_err}")


def cf_run(model: str, payload: dict, timeout: int = 300) -> dict:
    """Call the native /ai/run endpoint (used for Whisper), with the same rotation."""
    last = None
    for attempt in range(len(ACCOUNTS) + 3):
        idx = _current
        acc, tok = ACCOUNTS[idx]
        try:
            r = requests.post(
                f"https://api.cloudflare.com/client/v4/accounts/{acc}/ai/run/{model}",
                headers={"Authorization": f"Bearer {tok}"},
                json=payload,
                timeout=timeout,
            )
        except requests.RequestException as e:
            last = e
            time.sleep(min(2**attempt, 20))
            continue
        if r.ok and r.json().get("success", True):
            return r.json().get("result", r.json())
        last = f"{r.status_code} {r.text[:500]}"
        if r.status_code == 429 or is_quota_error(last):
            _advance(idx)
        else:
            time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"cf_run({model}) failed: {last}")


def vision(prompt: str, image_bytes: bytes, mime: str = "image/png", model: str = VISION_MODEL) -> str:
    b64 = base64.b64encode(image_bytes).decode()
    msg = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
    ]}]
    return invoke_rotating(msg, model=model).content


def transcribe(audio_bytes: bytes) -> str:
    res = cf_run(ASR_MODEL, {"audio": base64.b64encode(audio_bytes).decode()})
    return (res.get("text") or "").strip()
