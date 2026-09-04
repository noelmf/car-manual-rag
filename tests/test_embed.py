"""Batching and pacing exist because of the provider's limits, not ours."""

import pytest

from car_manual_rag import embed


class TestBatches:
    def test_splits_on_the_count_limit(self):
        assert [len(b) for b in embed.batches(["x"] * (embed.BATCH + 1))] == [embed.BATCH, 1]

    def test_splits_on_the_size_limit_before_the_count_one(self):
        # Real chunks are ~1,100 chars, so the size cap is what usually binds.
        big = "x" * (embed.MAX_CHARS // 2 + 1)
        assert [len(b) for b in embed.batches([big] * 3)] == [1, 1, 1]

    def test_everything_is_kept_and_in_order(self):
        texts = [str(i) for i in range(200)]
        assert [t for b in embed.batches(texts) for t in b] == texts

    def test_nothing_yields_nothing(self):
        assert list(embed.batches([])) == []


class TestPace:
    @pytest.fixture(autouse=True)
    def clock(self, monkeypatch):
        """A clock the test drives: sleeping moves it, nothing really waits."""
        self.now, self.slept = 1000.0, []

        def sleep(seconds):
            self.slept.append(seconds)
            self.now += seconds

        monkeypatch.setattr(embed, "_sent", type(embed._sent)())
        monkeypatch.setattr(embed.time, "sleep", sleep)
        monkeypatch.setattr(embed.time, "monotonic", lambda: self.now)

    def test_the_first_batch_never_waits(self):
        embed.pace(embed.RATE)
        assert self.slept == []

    def test_waits_until_the_budget_frees_up(self):
        embed.pace(embed.RATE)
        embed.pace(1)
        assert len(self.slept) == 1 and embed.WINDOW <= self.slept[0] <= embed.WINDOW + 1

    def test_batches_that_fit_the_budget_go_straight_out(self):
        for _ in range(embed.RATE // embed.BATCH):
            embed.pace(embed.BATCH)
        assert self.slept == []

    def test_it_settles_into_the_allowed_rate(self):
        for _ in range(embed.RATE // embed.BATCH * 3):
            embed.pace(embed.BATCH)
        sent = (embed.RATE // embed.BATCH * 3) * embed.BATCH
        elapsed = self.now - 1000.0
        assert sent / (elapsed / embed.WINDOW + 1) <= embed.RATE

    def test_the_batch_size_divides_the_budget(self):
        # At BATCH=50 and RATE=90 only one batch fits per window, so the
        # effective rate would be 50/min instead of 90.
        assert embed.RATE % embed.BATCH == 0


class TestEmbed:
    @pytest.fixture(autouse=True)
    def settings(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "clave")
        monkeypatch.setenv("GEMINI_EMBEDDING", "modelo-emb")
        monkeypatch.setattr(embed, "_sent", type(embed._sent)())
        monkeypatch.setattr(embed.time, "sleep", lambda s: None)

    def fake_api(self, monkeypatch):
        sent = []

        def call(model, verb, payload, note=None):
            sent.append((model, verb, payload))
            n = len(payload["requests"])
            return {"embeddings": [{"values": [float(i)]} for i in range(n)]}

        monkeypatch.setattr(embed, "call", call)
        return sent

    def test_a_passage_and_a_query_are_tagged_differently(self, monkeypatch):
        # Gemini answers differently to each; swapping them costs recall in
        # silence, so the mapping is worth pinning down.
        sent = self.fake_api(monkeypatch)
        embed.embed(["x"], "passage")
        embed.embed(["x"], "query")
        assert [s[2]["requests"][0]["taskType"] for s in sent] == [
            "RETRIEVAL_DOCUMENT",
            "RETRIEVAL_QUERY",
        ]

    def test_the_request_carries_the_model_and_the_dimensions(self, monkeypatch):
        sent = self.fake_api(monkeypatch)
        embed.embed(["x"], "passage")
        request = sent[0][2]["requests"][0]
        assert sent[0][0] == "modelo-emb"
        assert request["model"] == "models/modelo-emb"
        assert request["outputDimensionality"] == embed.DIMS

    def test_vectors_come_back_in_the_order_of_the_texts(self, monkeypatch):
        self.fake_api(monkeypatch)
        vectors = embed.embed([str(i) for i in range(embed.BATCH + 5)], "passage")
        assert len(vectors) == embed.BATCH + 5

    def test_an_unknown_kind_is_rejected_before_any_request(self, monkeypatch):
        sent = self.fake_api(monkeypatch)
        with pytest.raises(ValueError, match="passage"):
            embed.embed(["x"], "documento")
        assert sent == []
