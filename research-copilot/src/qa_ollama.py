# src/qa_ollama.py
from __future__ import annotations

from typing import List, Dict
import re
import requests

from config import LLM_MODEL


_CITATION_RE = re.compile(r"\[(\d+)\]")


def build_prompt(question: str, hits: List[Dict]) -> str:
    """
    Strict, citation-heavy prompt:
    - Requires every sentence to end with citations like [1] or [1][2]
    - Forces a fixed output structure
    - Forces "INSUFFICIENT_EVIDENCE" when not answerable from evidence
    """
    evidence_blocks = []
    for i, h in enumerate(hits, start=1):
        src = h["meta"].get("source", "unknown")
        page = h["meta"].get("page", "?")
        text = h["text"].strip()

        evidence_blocks.append(
            f"[{i}] SOURCE: {src} | page {page}\n"
            f"EVIDENCE:\n{text}\n"
        )

    evidence = "\n".join(evidence_blocks)

    return f"""You are Research Copilot.

You must answer using ONLY the EVIDENCE blocks below.
If the evidence does not contain the answer, output exactly:
INSUFFICIENT_EVIDENCE: <what is missing>

EVIDENCE:
{evidence}

QUESTION:
{question}

STRICT RULES (must follow):
- Do NOT use outside knowledge.
- Do NOT guess.
- Every sentence MUST end with citations like [1] or [1][2].
- Only cite evidence blocks that directly support that sentence.
- If you cannot cite a sentence, do not write it.

OUTPUT FORMAT (exact headings):
SIMPLE_EXPLANATION:
<2-4 sentences, each ends with citations>

TECHNICAL_EXPLANATION:
<2-4 sentences, each ends with citations>

KEY_EVIDENCE:
- <1 bullet quoting/paraphrasing the most relevant evidence> [#]
- <optional 2nd bullet> [#]
"""


def ollama_generate(prompt: str, model: str = LLM_MODEL) -> str:
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["response"].strip()


def _has_citations(text: str) -> bool:
    return bool(_CITATION_RE.search(text))


def _citations_within_range(text: str, max_id: int) -> bool:
    ids = [int(m.group(1)) for m in _CITATION_RE.finditer(text)]
    return all(1 <= i <= max_id for i in ids) if ids else False


def _every_nonempty_sentence_has_citation(text: str) -> bool:
    """
    Simple guard: split on sentence-ish endings and ensure each non-empty sentence has [n].
    Not perfect NLP, but works well in practice.
    """
    # Remove headings to avoid false negatives
    cleaned = re.sub(r"(?m)^(SIMPLE_EXPLANATION|TECHNICAL_EXPLANATION|KEY_EVIDENCE):\s*$", "", text).strip()
    # Split into sentences heuristically
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    for s in parts:
        s = s.strip()
        if not s:
            continue
        # Allow bullet lines as "sentences" too
        if not _CITATION_RE.search(s):
            return False
    return True


def answer(question: str, hits: List[Dict]) -> str:
    """
    Generate an answer with a lightweight citation guard:
    - If missing citations or citing out-of-range blocks, reprompt once with an even stricter reminder.
    """
    prompt = build_prompt(question, hits)
    out = ollama_generate(prompt, model=LLM_MODEL)

    max_id = len(hits)

    ok = (
        _has_citations(out)
        and _citations_within_range(out, max_id)
        and _every_nonempty_sentence_has_citation(out)
    )

    if ok:
        return out

    # One reprompt: tighten further + explicitly request compliance
    reprompt = prompt + """
REPAIR INSTRUCTIONS:
- Your previous answer violated the citation rules.
- Rewrite the entire answer strictly following the rules.
- Every sentence must end with citations like [1] or [1][2].
- If you cannot answer from evidence, output INSUFFICIENT_EVIDENCE.
"""
    out2 = ollama_generate(reprompt, model=LLM_MODEL)
    return out2
