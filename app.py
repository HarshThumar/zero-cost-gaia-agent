"""HF Space: log in with Hugging Face, run the agent on all questions, submit, show the score.

Adapted from the Agents Course Unit 4 template. Secrets (CF_AI_*, SERPER_API_KEY, HF_TOKEN) come
from Space secrets.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import gradio as gr
import pandas as pd
import requests

from agent.files import API_URL, get_attachment
from agent.graph import run
from vote import pick_answer

SPACE_ID = os.getenv("SPACE_ID", "Harsh5764/zero-cost-gaia-agent")


VOTES = int(os.getenv("AGENT_VOTES", "3"))  # self-consistency: run each question N times, majority vote


def answer_one(q: dict) -> dict:
    candidates = []
    for i in range(VOTES):
        try:
            path = get_attachment(q["task_id"], q.get("file_name"))
            ans, _ = run(q["question"], str(path) if path else None)
            candidates.append(ans)
        except Exception as e:  # noqa: BLE001
            print(f"[{q['task_id'][:8]}] attempt {i + 1} error: {e}")
    return {"task_id": q["task_id"], "question": q["question"],
            "submitted_answer": pick_answer(candidates), "candidates": " | ".join(candidates)}


def run_and_submit_all(profile: gr.OAuthProfile | None):
    if not profile:
        return "Please log in with the Hugging Face button first.", None
    username = profile.username
    agent_code = f"https://huggingface.co/spaces/{SPACE_ID}/tree/main"

    try:
        questions = requests.get(f"{API_URL}/questions", timeout=60).json()
    except Exception as e:  # noqa: BLE001
        return f"Could not fetch questions: {e}", None

    with ThreadPoolExecutor(3) as ex:
        results = list(ex.map(answer_one, questions))
    table = pd.DataFrame(results)
    answers = [{"task_id": r["task_id"], "submitted_answer": r["submitted_answer"]} for r in results]

    try:
        r = requests.post(f"{API_URL}/submit", json={"username": username, "agent_code": agent_code, "answers": answers}, timeout=120)
        r.raise_for_status()
        d = r.json()
        status = (f"Submission successful!\nUser: {d.get('username')}\n"
                  f"Score: {d.get('score')}% ({d.get('correct_count')}/{d.get('total_attempted')} correct)\n"
                  f"{d.get('message', '')}")
    except Exception as e:  # noqa: BLE001
        status = f"Submission failed: {e}"
    return status, table


with gr.Blocks(title="Zero-Cost GAIA Agent") as demo:
    gr.Markdown(
        "# Zero-Cost GAIA Agent (LangGraph + free Cloudflare Workers AI)\n"
        "A LangGraph tool-using agent: Serper web search, Wikipedia (with historical revisions), file readers, "
        "Whisper transcription, vision, YouTube transcript/frame analysis, a Python sandbox and Stockfish.\n\n"
        "1. Log in with Hugging Face. 2. Click **Run & Submit**. A full run takes several minutes."
    )
    gr.LoginButton()
    btn = gr.Button("Run Evaluation & Submit All Answers", variant="primary")
    status = gr.Textbox(label="Status / Score", lines=5, interactive=False)
    table = gr.DataFrame(label="Questions and agent answers", wrap=True)
    btn.click(run_and_submit_all, outputs=[status, table])

if __name__ == "__main__":
    demo.launch()
