"""FastAPI application exposing the Documentary Q&A backend.

Endpoints:
  POST /ask            -> {answer, sources}            (grounded answer + ranked sources)
  POST /ask?stream=true-> text/event-stream            (bonus: token-by-token streaming)
  GET  /health         -> {status, chunks, provider}
  GET  /               -> minimal web UI                (bonus)
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse

from .config import get_settings
from .llm import LLMClient
from .prompt import OUT_OF_SCOPE_ANSWER, build_messages
from .retriever import Retriever
from .schemas import AskRequest, AskResponse, Source

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


class QAService:
    """Builds the index once, then answers questions against it."""

    def __init__(self):
        self.settings = get_settings()
        self.retriever = Retriever(self.settings)
        self.llm = LLMClient(self.settings)

    def _sources(self, hits) -> list[Source]:
        return [
            Source(timestamp=h.chunk.timestamp, excerpt=h.chunk.excerpt, score=round(h.score, 4))
            for h in hits
        ]

    def answer(self, question: str) -> AskResponse:
        result = self.retriever.retrieve(question)
        if not result.in_scope:
            # Out-of-scope: never call the LLM, so it cannot hallucinate an answer.
            return AskResponse(answer=OUT_OF_SCOPE_ANSWER, sources=[])
        messages = build_messages(question, result.hits)
        answer = self.llm.complete(messages)
        return AskResponse(answer=answer, sources=self._sources(result.hits))

    def stream(self, question: str):
        """Yield Server-Sent-Events: token deltas, then a final 'sources' event."""
        result = self.retriever.retrieve(question)
        if not result.in_scope:
            yield _sse("token", {"text": OUT_OF_SCOPE_ANSWER})
            yield _sse("sources", {"sources": []})
            yield _sse("done", {})
            return
        messages = build_messages(question, result.hits)
        for delta in self.llm.stream(messages):
            yield _sse("token", {"text": delta})
        sources = [s.model_dump() for s in self._sources(result.hits)]
        yield _sse("sources", {"sources": sources})
        yield _sse("done", {})


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


service: QAService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    service = QAService()  # builds + embeds the index at startup
    yield


app = FastAPI(title="Documentary Q&A Backend", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    assert service is not None
    return {
        "status": "ok",
        "chunks": len(service.retriever.index),
        "llm_provider": service.settings.llm_provider,
        "llm_model": service.llm.model,
        "embed_backend": service.settings.embed_backend,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, stream: bool = Query(False)):
    assert service is not None
    try:
        if stream:
            return StreamingResponse(service.stream(req.question), media_type="text/event-stream")
        return service.answer(req.question)
    except Exception as exc:  # surface provider/connectivity errors cleanly
        raise HTTPException(status_code=503, detail=f"LLM backend error: {exc}") from exc


@app.get("/")
def index_page():
    html = STATIC_DIR / "index.html"
    if html.exists():
        return FileResponse(html)
    return {"message": "Documentary Q&A backend. POST /ask with {\"question\": \"...\"}."}
