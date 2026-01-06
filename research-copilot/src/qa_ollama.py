# src/qa_ollama.py
from __future__ import annotations

from typing import List, Dict, Iterator
import re

from src.llm_client import generate, generate_stream


_CITATION_RE = re.compile(r"\[(\d+)\]")


def build_prompt(
    question: str,
    hits: List[Dict],
    chat_history: List[Dict] | None = None,
) -> str:
    """
    Strict, citation-heavy prompt:
    - Requires every sentence to end with citations like [1] or [1][2]
    - Forces a fixed output structure
    - Forces "INSUFFICIENT_EVIDENCE" when not answerable from evidence
    """

    history_blocks = []
    for m in (chat_history or [])[-8:]:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        prefix = "ASSISTANT" if role == "assistant" else "USER"
        history_blocks.append(f"{prefix}: {content}")

    history = "\n".join(history_blocks) if history_blocks else "None"

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
CHAT HISTORY is provided for conversational context ONLY — it is NOT evidence.
Do NOT cite CHAT HISTORY. Citations must ONLY refer to EVIDENCE blocks.

CHAT HISTORY:
{history}

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


def _has_citations(text: str) -> bool:
    return bool(_CITATION_RE.search(text))


def _citations_within_range(text: str, max_id: int) -> bool:
    ids = [int(m.group(1)) for m in _CITATION_RE.finditer(text)]
    return all(1 <= i <= max_id for i in ids) if ids else False


def _every_nonempty_sentence_has_citation(text: str) -> bool:
    """
    Simple guard: split on sentence-ish endings and ensure each non-empty sentence has [n].
    """
    cleaned = re.sub(
        r"(?m)^(SIMPLE_EXPLANATION|TECHNICAL_EXPLANATION|KEY_EVIDENCE):\s*$",
        "",
        text,
    ).strip()

    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    for s in parts:
        s = s.strip()
        if not s:
            continue
        if not _CITATION_RE.search(s):
            return False
    return True


def answer(
    question: str,
    hits: List[Dict],
    chat_history: List[Dict] | None = None,
) -> str:
    """
    Generate an answer with a citation guard.
    If the first attempt violates citation rules, retry once.
    """
    prompt = build_prompt(question, hits, chat_history=chat_history)
    out = generate(prompt)

    max_id = len(hits)

    ok = (
        _has_citations(out)
        and _citations_within_range(out, max_id)
        and _every_nonempty_sentence_has_citation(out)
    )

    if ok:
        return out

    reprompt = prompt + """
REPAIR INSTRUCTIONS:
- Your previous answer violated the citation rules.
- Rewrite the entire answer strictly following the rules.
- Every sentence must end with citations like [1] or [1][2].
- If you cannot answer from evidence, output INSUFFICIENT_EVIDENCE.
"""
    return generate(reprompt)


def answer_stream(
    question: str,
    hits: List[Dict],
    chat_history: List[Dict] | None = None,
) -> Iterator[str]:
    """
    Stream an answer (tokens).
    Citation validation is applied only in non-streaming mode.
    """
    prompt = build_prompt(question, hits, chat_history=chat_history)
    yield from generate_stream(prompt)
