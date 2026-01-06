# src/llm_client.py
from __future__ import annotations
from typing import Iterator
import os

from groq import Groq

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is not set")

_client = Groq(api_key=GROQ_API_KEY)


def generate(prompt: str) -> str:
    """
    Non-streaming text generation.
    """
    completion = _client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )
    return completion.choices[0].message.content.strip()


def generate_stream(prompt: str) -> Iterator[str]:
    """
    Streaming text generation.
    Yields tokens.
    """
    stream = _client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content
