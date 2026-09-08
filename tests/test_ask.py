"""The prompt and the reply parsing are the whole of the generation side."""

import pytest

from car_manual_rag import ask, config


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    # The developer's own .env must not reach these tests. load_env() only
    # fills in what is missing, so a variable a test wants absent would be put
    # back from the file; marking the file as already read stops that.
    monkeypatch.setattr(config, "_loaded", True)
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


class TestThinkingBudget:
    """GEMINI_THINKING is optional, and sent only when it is set."""

    def test_nothing_is_sent_when_it_is_unset(self, monkeypatch):
        monkeypatch.delenv("GEMINI_THINKING", raising=False)
        config = ask.payload_for("¿cuando?", HITS)["generationConfig"]
        assert "thinkingConfig" not in config

    def test_the_level_is_sent_when_it_is_set(self, monkeypatch):
        monkeypatch.setenv("GEMINI_THINKING", "low")
        config = ask.payload_for("¿cuando?", HITS)["generationConfig"]
        assert config["thinkingConfig"] == {"thinkingLevel": "low"}

    def test_an_empty_value_counts_as_unset(self, monkeypatch):
        # A model that never heard of the field rejects the whole request, so
        # an empty line in .env must not turn into one.
        monkeypatch.setenv("GEMINI_THINKING", "")
        assert "thinkingConfig" not in ask.payload_for("¿cuando?", HITS)["generationConfig"]

    def test_the_answer_is_still_pinned_to_the_manual(self, monkeypatch):
        monkeypatch.setenv("GEMINI_THINKING", "low")
        config = ask.payload_for("¿cuando?", HITS)["generationConfig"]
        assert config["temperature"] == 0 and config["maxOutputTokens"] == ask.MAX_TOKENS


class TestStreaming:
    """Three events: hits once, token many times, then done or error."""

    def run(self, monkeypatch, chunks):
        monkeypatch.setattr(ask, "search", lambda *a, **k: HITS)
        monkeypatch.setattr(ask, "stream", lambda *a, **k: iter(chunks))
        return list(ask.ask_stream("M", "¿cuando?"))

    def said(self, text, finish=None):
        candidate = {"content": {"parts": [{"text": text}]}}
        if finish:
            candidate["finishReason"] = finish
        return {"candidates": [candidate], "usageMetadata": {"totalTokenCount": 9}}

    def test_the_fragments_come_first_so_there_is_something_to_read(self, monkeypatch):
        events = self.run(monkeypatch, [self.said("Cada", "STOP")])
        assert events[0]["event"] == "hits"
        assert events[0]["hits"] == HITS

    def test_each_piece_of_prose_is_its_own_event(self, monkeypatch):
        events = self.run(monkeypatch, [self.said("Cada "), self.said("dos anos.", "STOP")])
        assert [e["text"] for e in events if e["event"] == "token"] == ["Cada ", "dos anos."]

    def test_a_finished_answer_ends_with_done(self, monkeypatch):
        events = self.run(monkeypatch, [self.said("Vale.", "STOP")])
        assert events[-1]["event"] == "done"
        assert events[-1]["tokens"] == 9

    def test_an_answer_cut_off_ends_with_error_not_done(self, monkeypatch):
        # The reader has already seen the beginning; the only honest thing left
        # is to say the rest never came.
        events = self.run(monkeypatch, [self.said("Aparque el vehic", "MAX_TOKENS")])
        assert events[-1]["event"] == "error"
        assert "MAX_TOKENS" in events[-1]["error"]
        assert not any(e["event"] == "done" for e in events)

    def test_a_stream_that_never_says_why_it_stopped_is_an_error(self, monkeypatch):
        events = self.run(monkeypatch, [self.said("A medias")])
        assert events[-1]["event"] == "error"

    def test_an_empty_chunk_yields_no_token(self, monkeypatch):
        events = self.run(monkeypatch, [self.said(""), self.said("Vale.", "STOP")])
        assert len([e for e in events if e["event"] == "token"]) == 1


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
