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
    def test_carries_the_fragment_text_and_its_section(self):
        out = ask.prompt("¿cuando?", HITS)
        assert "El liquido de frenos." in out and "Frenos" in out

    def test_the_question_comes_last(self):
        out = ask.prompt("¿cuando?", HITS)
        assert out.rindex("¿cuando?") > out.rindex("El liquido")

    def test_fragments_are_numbered_from_one(self):
        assert "Fragment 1" in ask.prompt("x", HITS)

    def test_the_fragment_header_carries_the_section_but_never_a_page(self):
        out = ask.prompt("¿cuando?", HITS)
        assert "Fragment 1 (Frenos)" in out
        assert "pag." not in out

    def test_the_system_prompt_forbids_headings_and_preamble(self):
        assert "No headings" in ask.SYSTEM
        assert "begin with the answer itself" in ask.SYSTEM

    def test_the_system_prompt_forbids_sending_the_reader_to_a_page(self):
        # The model is told not to cite because nothing shows it a page to cite.
        assert "Do not send the reader anywhere" in ask.SYSTEM
        assert "(pag. N)" not in ask.SYSTEM


class TestSeen:
    """The chunker repeats a paragraph across the seam between two chunks."""

    def hits(self, *texts):
        return [dict(HITS[0], text=t) for t in texts]

    def test_a_paragraph_already_shown_is_not_shown_again(self):
        out = list(ask.seen(self.hits("uno\ndos", "dos\ntres")))
        assert [h["text"] for h in out] == ["uno\ndos", "tres"]

    def test_a_hit_that_adds_nothing_new_is_dropped_entirely(self):
        out = list(ask.seen(self.hits("uno\ndos", "dos")))
        assert [h["text"] for h in out] == ["uno\ndos"]

    def test_hits_that_share_nothing_all_survive(self):
        out = list(ask.seen(self.hits("uno", "dos")))
        assert [h["text"] for h in out] == ["uno", "dos"]

    def test_the_rest_of_the_hit_is_carried_through(self):
        out = list(ask.seen(self.hits("uno")))
        assert out[0]["score"] == HITS[0]["score"] and out[0]["section"] == "Frenos"


class TestUnfinished:
    """A reasoning model spends the token ceiling before it writes a word."""

    def answer(self, monkeypatch, reply):
        monkeypatch.setattr(ask, "search", lambda *a, **k: HITS)
        monkeypatch.setattr(ask, "call", lambda *a, **k: reply)
        return ask.ask("M", "¿cuando?")

    def truncated(self, reason):
        return {
            "candidates": [
                {"content": {"parts": [{"text": "Aparque el vehic"}]}, "finishReason": reason}
            ]
        }

    def test_an_answer_cut_off_by_the_ceiling_is_refused(self, monkeypatch):
        with pytest.raises(RuntimeError, match="MAX_TOKENS"):
            self.answer(monkeypatch, self.truncated("MAX_TOKENS"))

    def test_the_refusal_names_the_ceiling_that_has_to_change(self, monkeypatch):
        with pytest.raises(RuntimeError, match=str(ask.MAX_TOKENS)):
            self.answer(monkeypatch, self.truncated("MAX_TOKENS"))

    def test_any_other_early_stop_is_refused_too(self, monkeypatch):
        with pytest.raises(RuntimeError, match="SAFETY"):
            self.answer(monkeypatch, self.truncated("SAFETY"))

    def test_a_finished_answer_passes(self, monkeypatch):
        out = self.answer(monkeypatch, self.truncated("STOP"))
        assert out["answer"] == "Aparque el vehic"

    def test_a_reply_without_a_reason_is_not_treated_as_truncated(self, monkeypatch):
        out = self.answer(
            monkeypatch, {"candidates": [{"content": {"parts": [{"text": "Vale."}]}}]}
        )
        assert out["answer"] == "Vale."


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
