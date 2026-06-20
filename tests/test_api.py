"""API tests. The happy path stubs the LLM so the suite runs without any provider;
the out-of-scope path needs no LLM at all (it short-circuits before calling one)."""
import app.main as main_module
from app.prompt import OUT_OF_SCOPE_ANSWER
from fastapi.testclient import TestClient
from app.main import app


def test_health_and_ask_flows():
    with TestClient(app) as client:
        # Stub the LLM so the happy path does not require a running provider.
        main_module.service.llm.complete = lambda messages: "Stubbed grounded answer [1]."

        health = client.get("/health").json()
        assert health["status"] == "ok"
        assert health["chunks"] > 0

        # In-scope question -> answer + ranked sources with timestamp + excerpt.
        r = client.post("/ask", json={"question": "What did the Victorians add to bread?"})
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] == "Stubbed grounded answer [1]."
        assert 2 <= len(body["sources"]) <= 3
        for src in body["sources"]:
            assert len(src["timestamp"]) == 8 and src["excerpt"]

        # Out-of-scope -> canned abstention, no sources, LLM never invoked.
        r2 = client.post("/ask", json={"question": "What is the capital of Australia?"})
        body2 = r2.json()
        assert body2["answer"] == OUT_OF_SCOPE_ANSWER
        assert body2["sources"] == []


def test_ask_validates_empty_question():
    with TestClient(app) as client:
        # Empty string (min_length) and whitespace-only (validator) are both rejected.
        assert client.post("/ask", json={"question": ""}).status_code == 422
        assert client.post("/ask", json={"question": "   "}).status_code == 422


def test_stream_emits_tokens_then_sources_and_done():
    with TestClient(app) as client:
        main_module.service.llm.stream = lambda messages: iter(["Hello ", "world [1]."])

        with client.stream("POST", "/ask?stream=true",
                           json={"question": "What did the Victorians add to bread?"}) as r:
            events = []
            for line in r.iter_lines():
                if line.startswith("event:"):
                    events.append(line.split(":", 1)[1].strip())
        assert "token" in events and "sources" in events and events[-1] == "done"


def test_stream_emits_error_frame_on_llm_failure():
    with TestClient(app) as client:
        def boom(messages):
            raise RuntimeError("provider down")
            yield  # make it a generator

        main_module.service.llm.stream = boom
        with client.stream("POST", "/ask?stream=true",
                           json={"question": "What did the Victorians add to bread?"}) as r:
            events = [line.split(":", 1)[1].strip()
                      for line in r.iter_lines() if line.startswith("event:")]
        # Mid-stream failure surfaces as an explicit error frame, then done.
        assert "error" in events and events[-1] == "done"


def test_stream_emits_error_frame_on_retrieval_failure():
    with TestClient(app) as client:
        main_module.service.retriever.retrieve = lambda question: (_ for _ in ()).throw(RuntimeError("index down"))
        with client.stream("POST", "/ask?stream=true",
                           json={"question": "Who is Annie Gray?"}) as r:
            events = [line.split(":", 1)[1].strip()
                      for line in r.iter_lines() if line.startswith("event:")]
        assert "error" in events and events[-1] == "done"


def test_ask_maps_retrieval_failure_to_503():
    with TestClient(app) as client:
        main_module.service.retriever.retrieve = lambda question: (_ for _ in ()).throw(RuntimeError("index down"))
        r = client.post("/ask", json={"question": "Who is Annie Gray?"})
        assert r.status_code == 503
        assert "Retrieval backend error" in r.json()["detail"]


def test_empty_llm_completion_abstains_without_sources():
    with TestClient(app) as client:
        main_module.service.llm.complete = lambda messages: ""
        r = client.post("/ask", json={"question": "Who is Annie Gray?"})
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] == OUT_OF_SCOPE_ANSWER
        assert body["sources"] == []


def test_stream_empty_llm_output_abstains_without_sources():
    with TestClient(app) as client:
        main_module.service.llm.stream = lambda messages: iter([])
        with client.stream("POST", "/ask?stream=true",
                           json={"question": "Who is Annie Gray?"}) as r:
            lines = list(r.iter_lines())
        assert any("isn't covered in the documentary" in line for line in lines if line.startswith("data:"))
        assert any('"sources": []' in line for line in lines if line.startswith("data:"))
