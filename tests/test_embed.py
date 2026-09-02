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
