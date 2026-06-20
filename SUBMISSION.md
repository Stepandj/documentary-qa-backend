# Submission summary

I'm particularly happy with the **hybrid dense + BM25 retrieval**: pure semantic search
missed exact-name questions (e.g. "Who is Annie Gray?", whose chunk is dominated by its
bread-tasting topic), and fusing a BM25 lexical signal via Reciprocal Rank Fusion fixed
the named-entity category at negligible cost — with a guard that ignores the lexical
signal when it's empty so it never injects spurious sources. **Out-of-scope handling is
deliberately two-layered**: a similarity-score threshold (calibrated on this transcript's
score distribution) short-circuits clearly off-topic questions *before any LLM call*, and
a strict abstention prompt is the backstop for topically-adjacent ones — so the service
says "I don't know" instead of hallucinating. The **LLM backend is fully provider-agnostic**
through the OpenAI-compatible API: it runs fully offline on Ollama by default and switches
to Kimi K2 / GLM / MiniMax / Groq / Gemini by environment variable alone, with no code
change. Chunking leans on the transcript's own `HH:MM:SS` markers, which give accurate
citation timestamps for free.

**Given more time** I'd calibrate the similarity threshold from a held-out in/out-of-scope
question set rather than by hand, quantitatively evaluate whether the `nomic-embed-text`
embedder and the optional cross-encoder reranker earn their latency, trim cited excerpts to
the supporting sentence, and add an answer-level grounding check that verifies each `[n]`
citation actually maps to a used source.
