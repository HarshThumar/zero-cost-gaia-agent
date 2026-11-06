"""LangGraph agent: plan → (LLM ⇄ tools)* → finalize → normalized answer."""

from __future__ import annotations

import datetime as dt
import json
import re
import time
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agent import llm
from agent.answer import extract_final, normalize
from agent.tools import ALL_TOOLS

MAX_STEPS = 25
TIME_BUDGET_S = 480  # per question; after this the agent must answer with what it has

SYSTEM_PROMPT = f"""You are a meticulous research agent answering questions from the GAIA benchmark.
Today's date is {dt.date.today().isoformat()}.

How to work:
- Think about what the question really asks, including every constraint (dates, units, "first name only",
  ordering, "as of", version of a source). Re-read the question before answering.
- Use tools to find and VERIFY facts; never guess from memory when a tool can check it.
  Prefer primary sources (Wikipedia article, official pages, the actual paper) over snippets.
- For Wikipedia, use wikipedia_page (it renders tables). If the question names a version/year of
  Wikipedia, pass as_of_date (e.g. "the 2022 version" → 2022-12-31).
- Do arithmetic, counting, sorting and data analysis with run_python, not in your head.
- Attachments are given as local file paths: read them with the matching tool.
- If a search approach fails, reformulate (different keywords, a different source) rather than giving up.
- If a site refuses access (403/blocked), do not guess from search snippets: find the same data on
  another source that allows access (Wikipedia, official league/organization sites, other databases).
- Some questions are puzzles needing no tools (e.g. reversed text, logic tables): solve them carefully.
- Chess board images: use chess_image_to_fen (pass the side to move), sanity-check the board, then chess_best_move.
- Videos: youtube_transcript for what is said, youtube_frames for what is seen.
- When the answer is a list of items taken from a source (recipe, transcript, document, table), copy each
  item's wording as the source states it (keep descriptive words) unless the question says to simplify.
- Before the final answer, re-check it against every format instruction in the question
  (ordering such as alphabetical, separators, units, rounding, singular/plural, which name part).

Answer format — your final message MUST end with:
FINAL ANSWER: [YOUR FINAL ANSWER]
YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
- Number: no thousands commas, no units ($, %, etc.) unless the question asks for them.
- String: no articles, no abbreviations (e.g. write city names in full), digits as digits unless told otherwise.
- List: comma separated, apply the rules above to each element; follow any ordering the question requests.
- Follow the question's own format instructions exactly (e.g. "algebraic notation", "IOC code", "two decimal places").
Never answer "unknown"; give your single best answer.
"""

FINALIZE_PROMPT = """Stop using tools. Based on everything above, give your best answer now.
Reply with one line exactly in the form:
FINAL ANSWER: [YOUR FINAL ANSWER]"""


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    steps: int
    started: float


TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}


def _repair_tool_calls(msg: AIMessage) -> tuple[AIMessage, list[ToolMessage]]:
    """gpt-oss sometimes emits raw text (e.g. Python code) instead of JSON tool arguments.
    If the tool takes one required string arg, wrap the raw text into it. Otherwise, reply
    with an error ToolMessage so the model can retry with valid arguments."""
    if not msg.invalid_tool_calls:
        return msg, []
    fixed, errors = list(msg.tool_calls), []
    for itc in msg.invalid_tool_calls:
        tool_ = TOOLS_BY_NAME.get(itc.get("name") or "")
        required = (tool_.args_schema.model_json_schema().get("required") or []) if tool_ else []
        raw = itc.get("args")
        if tool_ and isinstance(raw, str):
            # 1) JSON with raw newlines/control chars inside strings → lenient parse
            try:
                parsed = json.loads(raw, strict=False)
                if isinstance(parsed, dict) and set(required) <= set(parsed):
                    fixed.append({"name": tool_.name, "args": parsed, "id": itc["id"], "type": "tool_call"})
                    continue
            except ValueError:
                pass
        if tool_ and len(required) == 1 and isinstance(raw, str) and not raw.lstrip().startswith("{"):
            # 2) bare text (e.g. Python code) for a single-argument tool
            fixed.append({"name": tool_.name, "args": {required[0]: itc["args"]}, "id": itc["id"], "type": "tool_call"})
        else:
            errors.append(ToolMessage(
                f"Error: invalid tool call (arguments must be a JSON object matching the schema). Got: {str(itc.get('args'))[:300]}",
                tool_call_id=itc["id"], name=itc.get("name") or "unknown"))
    if errors:  # keep everything as-is so ids pair up with the error messages
        return msg, errors
    return msg.model_copy(update={"tool_calls": fixed, "invalid_tool_calls": []}), []


_LEAK_MARKERS = ("<|call|>", "to=functions", "<|start|>", "<|channel|>")
NUDGE = ("Your previous reply contained a tool call written as plain text, which was not executed. "
         "Call tools only through the function-calling interface, with exact JSON arguments.")


def _leaked_tool_call(msg: AIMessage) -> bool:
    """gpt-oss sometimes writes its tool call into the text (harmony tokens) instead of tool_calls."""
    return not msg.tool_calls and not msg.invalid_tool_calls and any(m in (msg.content or "") for m in _LEAK_MARKERS)


def agent_node(state: State) -> dict:
    msg = llm.invoke_rotating(state["messages"], tools=ALL_TOOLS)
    for attempt in range(3):  # resample instead of letting a garbled call reach finalize
        if not _leaked_tool_call(msg):
            break
        msg = llm.invoke_rotating(state["messages"], tools=ALL_TOOLS, temperature=0.4 + 0.2 * attempt)
    extra: list[AnyMessage] = []
    if _leaked_tool_call(msg):
        msg = msg.model_copy(update={"content": "(malformed tool call omitted)"})
        extra.append(HumanMessage(NUDGE))
    msg, errors = _repair_tool_calls(msg)
    return {"messages": [msg, *errors, *extra], "steps": state.get("steps", 0) + 1}


def finalize_node(state: State) -> dict:
    msgs = state["messages"]
    # Drop a dangling tool-call request so the conversation stays valid without tools.
    if isinstance(msgs[-1], AIMessage) and (msgs[-1].tool_calls or msgs[-1].invalid_tool_calls):
        msgs = msgs[:-1]
    msg = llm.invoke_rotating(msgs + [HumanMessage(FINALIZE_PROMPT)])
    return {"messages": [msg]}


FORMAT_CHECK_PROMPT = """You are checking the FORMAT of an answer to a benchmark question. Scoring is exact match.

Question:
{question}

Proposed answer: {answer}

Check the proposed answer against every explicit format instruction in the question (ordering such as
alphabetical, separators, units/currency, decimal places, rounding, which part of a name, abbreviation
style, singular/plural, notation) and the general rules: no units unless asked, no thousands commas,
no trailing period, lists comma separated.
Fix ONLY formatting. Do not change facts, add or remove items, or re-answer the question.
Reply with exactly one line: FINAL ANSWER: <answer>"""


def format_check_node(state: State) -> dict:
    question = next(m.content for m in state["messages"] if isinstance(m, HumanMessage))
    proposed = normalize(extract_final(state["messages"][-1].content or ""))
    if not proposed:
        return {}
    try:
        msg = llm.invoke_rotating([HumanMessage(FORMAT_CHECK_PROMPT.format(question=question, answer=proposed))])
    except Exception:  # noqa: BLE001 — a failed check must never lose the answer
        return {}
    checked = normalize(extract_final(msg.content or ""))
    if not checked or not _format_only_change(proposed, checked):
        return {}  # the checker tried to change content → keep the agent's answer
    return {"messages": [AIMessage(f"(format check) FINAL ANSWER: {checked}")]}


def _words(s: str) -> list[str]:
    return sorted(re.findall(r"[^\W_]+", s.lower()))


def _format_only_change(before: str, after: str) -> bool:
    """Accept reorderings / punctuation / dropping at most one token (a unit, an article).
    Reject anything that adds new words or changes values."""
    b, a = _words(before), _words(after)
    if a == b:
        return True
    extra = list(a)
    for w in b:
        if w in extra:
            extra.remove(w)
    return not extra and len(b) - len(a) <= 1


def _exhausted(state: State) -> bool:
    return state.get("steps", 0) >= MAX_STEPS or time.time() - state.get("started", time.time()) > TIME_BUDGET_S


def route(state: State) -> str:
    last = state["messages"][-1]
    if isinstance(last, (ToolMessage, HumanMessage)):  # error feedback / nudge → let the model retry
        return "finalize" if _exhausted(state) else "agent"
    if isinstance(last, AIMessage) and last.tool_calls:
        return "finalize" if _exhausted(state) else "tools"
    if "final answer" in (last.content or "").lower():
        return "format_check"
    return "finalize"


def build_graph():
    g = StateGraph(State)
    g.add_node("agent", agent_node)
    g.add_node("tools", ToolNode(ALL_TOOLS, handle_tool_errors=True))
    g.add_node("finalize", finalize_node)
    g.add_node("format_check", format_check_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", "agent": "agent", "finalize": "finalize",
                                             "format_check": "format_check"})
    g.add_edge("tools", "agent")
    g.add_edge("finalize", "format_check")
    g.add_edge("format_check", END)
    return g.compile()


GRAPH = build_graph()


def run(question: str, file_path: str | None = None) -> tuple[str, list[AnyMessage]]:
    """Answer one question. Returns (normalized answer, full message trace)."""
    content = question
    if file_path:
        content += f"\n\nAttached file (local path): {file_path}"
    state = GRAPH.invoke(
        {"messages": [SystemMessage(SYSTEM_PROMPT), HumanMessage(content)], "steps": 0, "started": time.time()},
        {"recursion_limit": MAX_STEPS * 2 + 10},
    )
    final = state["messages"][-1].content or ""
    return normalize(extract_final(final)), state["messages"]


def trace_markdown(messages: list[AnyMessage]) -> str:
    out = []
    for m in messages:
        if isinstance(m, SystemMessage):
            continue
        if isinstance(m, HumanMessage):
            out.append(f"### 👤 User\n{m.content}")
        elif isinstance(m, AIMessage):
            reasoning = (m.additional_kwargs or {}).get("reasoning_content")
            s = "### 🤖 Agent\n"
            if reasoning:
                s += f"<details><summary>reasoning</summary>\n\n{reasoning}\n</details>\n\n"
            s += m.content or ""
            for tc in m.tool_calls:
                s += f"\n→ `{tc['name']}({tc['args']})`"
            out.append(s)
        elif isinstance(m, ToolMessage):
            c = str(m.content)
            out.append(f"### 🔧 {m.name}\n```\n{c[:3000]}{'…' if len(c) > 3000 else ''}\n```")
    return "\n\n".join(out)
