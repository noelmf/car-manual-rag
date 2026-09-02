"""The prompt and the reply parsing are the whole of the generation side."""

import pytest

from car_manual_rag import ask


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    for name, value in [
        ("GEMINI_API_KEY", "clave"),
        ("GEMINI_EMBEDDING", "emb"),
        ("GEMINI_MODEL", "modelo"),
    ]:
        monkeypatch.setenv(name, value)


HITS = [
    {
        "section": "Frenos",
        "printed": ["10"],
        "pages": [2],
        "text": "El liquido de frenos.",
        "chunk_id": "M:0",
        "score": 0.8,
    }
]


class TestPrompt:
    def test_carries_the_fragment_text_and_its_citation(self):
        out = ask.prompt("¿cuando?", HITS)
        assert "El liquido de frenos." in out and "Frenos, pag. 10" in out

    def test_the_question_comes_last(self):
        out = ask.prompt("¿cuando?", HITS)
        assert out.rindex("¿cuando?") > out.rindex("El liquido")

    def test_fragments_are_numbered_from_one(self):
        assert "Fragment 1" in ask.prompt("x", HITS)

    def test_the_system_prompt_names_the_citation_format_cite_produces(self):
        # ask.SYSTEM and index.cite hold two halves of one contract.
        from car_manual_rag.index import cite

        assert "(pag. N)" in ask.SYSTEM
        assert cite(HITS[0]).startswith("Frenos, pag.")


class TestReply:
    def answer(self, monkeypatch, reply):
        monkeypatch.setattr(ask, "search", lambda *a, **k: HITS)
        monkeypatch.setattr(ask, "call", lambda *a, **k: reply)
        return ask.ask("M", "¿cuando?")

    def test_reads_the_text_out_of_the_candidate(self, monkeypatch):
        out = self.answer(
            monkeypatch, {"candidates": [{"content": {"parts": [{"text": "Cada 2 anos."}]}}]}
        )
        assert out["answer"] == "Cada 2 anos."

    def test_joins_several_parts(self, monkeypatch):
        out = self.answer(
            monkeypatch, {"candidates": [{"content": {"parts": [{"text": "a"}, {"text": "b"}]}}]}
        )
        assert out["answer"] == "ab"

    def test_a_blocked_reply_says_why_instead_of_returning_nothing(self, monkeypatch):
        with pytest.raises(RuntimeError, match="SAFETY"):
            self.answer(monkeypatch, {"promptFeedback": {"blockReason": "SAFETY"}})

    def test_the_hits_come_back_with_the_answer(self, monkeypatch):
        out = self.answer(monkeypatch, {"candidates": [{"content": {"parts": [{"text": "x"}]}}]})
        assert out["hits"] == HITS and out["model"] == "modelo"
