"""Prompt construction for grounded, cited answers.

Two-part grounding strategy (the threshold in the retriever is the other part):
  - The system prompt forbids using outside knowledge and *requires* an explicit
    "I don't know" when the sources don't contain the answer.
  - Sources are passed numbered, with their time codes, and the model is told to cite
    them as [1], [2] so answers stay traceable.
"""
from __future__ import annotations

from .index import SearchHit

SYSTEM_PROMPT = (
    "You are a careful assistant that answers questions about a single documentary, "
    "using ONLY the numbered transcript excerpts provided in the user message.\n"
    "Rules:\n"
    "1. Base every statement strictly on the excerpts. Do not use outside knowledge.\n"
    "2. If the excerpts do not contain the answer, reply exactly: "
    "\"I don't know — this isn't covered in the documentary.\" Do not guess.\n"
    "3. Cite the excerpts you use with their bracket numbers, e.g. [1], [2].\n"
    "4. Be concise and factual; answer in natural prose, not bullet points unless helpful."
)

# Message shown to the user when retrieval found nothing above the threshold. Returned
# without calling the LLM at all, so an out-of-scope question can never hallucinate.
OUT_OF_SCOPE_ANSWER = "I don't know — this isn't covered in the documentary."


def build_sources_block(hits: list[SearchHit]) -> str:
    lines = []
    for i, hit in enumerate(hits, start=1):
        lines.append(f"[{i}] (time {hit.chunk.timestamp}) {hit.chunk.text}")
    return "\n\n".join(lines)


def build_user_prompt(question: str, hits: list[SearchHit]) -> str:
    sources = build_sources_block(hits)
    return (
        f"Transcript excerpts:\n{sources}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the excerpts above, and cite the ones you use by number."
    )


def build_messages(question: str, hits: list[SearchHit]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question, hits)},
    ]
