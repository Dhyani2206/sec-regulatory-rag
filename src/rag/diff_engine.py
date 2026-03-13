import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import difflib
from dotenv import load_dotenv

from rag.config import RAGConfig
from llm_prompts import SYSTEM_PROMPT

load_dotenv()

def load_section_text(processed_dir: str, company: str, form_folder: str, year: str, section: str) -> Tuple[str, str]:
    """
    Returns (text, source_file) for the best matching processed JSON.
    Assumes filename contains year (you already have that pattern).
    """
    root = Path(processed_dir) / company.upper() / form_folder.lower()
    if not root.exists():
        raise FileNotFoundError(f"Missing folder: {root}")

    candidates = list(root.glob(f"*{year}*.json"))
    if not candidates:
        # fallback: any file, then pick later (still safe to fail)
        candidates = list(root.glob("*.json"))

    best_text = ""
    best_src = ""
    for fp in candidates:
        data = json.loads(fp.read_text(encoding="utf-8", errors="ignore"))
        sections = data.get("sections", {})
        obj = sections.get(section.upper())
        if isinstance(obj, dict) and isinstance(obj.get("text"), str):
            t = obj["text"].strip()
            if len(t) > len(best_text):
                best_text = t
                best_src = data.get("source_file", str(fp))

    if not best_text:
        raise ValueError(f"Section {section} not found for {company} {form_folder} {year}")
    return best_text, best_src

def make_diff(old: str, new: str) -> str:
    old_lines = [l.strip() for l in old.splitlines() if l.strip()]
    new_lines = [l.strip() for l in new.splitlines() if l.strip()]
    diff = difflib.unified_diff(old_lines, new_lines, lineterm="", fromfile="prior", tofile="current")
    return "\n".join(diff)

def summarize_diff(company: str, section: str, year_old: str, year_new: str, diff_text: str) -> str:
    from openai import OpenAI
    cfg = RAGConfig()
    client = OpenAI()

    prompt = f"""You are analyzing changes in SEC filing text.
Use ONLY the provided diff. Do not guess reasons or add requirements.
Task: summarize the most material additions/removals and categorize them into 3–6 bullets.
If the diff is too large, focus on new risk themes or materially changed wording.

Context:
Company: {company}
Section: {section}
From year: {year_old}
To year: {year_new}

Diff:
{diff_text}
"""

    resp = client.chat.completions.create(
        model=cfg.answer_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content

if __name__ == "__main__":
    cfg = RAGConfig()
    company = input("Company (e.g., AAPL): ").strip().upper()
    section = "ITEM 1A"
    year_new = input("Current year (e.g., 2025): ").strip()
    year_old = str(int(year_new) - 1)

    new_text, _ = load_section_text(cfg.processed_dir, company, "10-k", year_new, section)
    old_text, _ = load_section_text(cfg.processed_dir, company, "10-k", year_old, section)

    diff_text = make_diff(old_text, new_text)
    print(summarize_diff(company, section, year_old, year_new, diff_text))