"""Tests for the two search decisions: adaptive pool size and source diversity."""

import numpy as np
import pytest

from matching.search import (
    SearchConfig,
    complexity_score,
    cosine_similarity,
    diversify,
    pool_size_for,
)


class TestComplexityScore:
    def test_empty_is_zero(self):
        assert complexity_score("") == 0.0
        assert complexity_score("   ") == 0.0

    def test_short_generic_line_scores_low(self):
        assert complexity_score("It works.") < 0.2

    def test_long_specific_line_scores_high(self):
        text = "In late March the Northern Valley Transit Authority published its findings."
        assert complexity_score(text) > 0.5

    def test_score_is_bounded(self):
        long_text = " ".join(["Extraordinarily"] * 60)
        assert 0.0 <= complexity_score(long_text) <= 1.0

    def test_word_factor_saturates_at_fifteen_words(self):
        # Past the cap, adding words must not raise the score further.
        fifteen = " ".join(["word"] * 15)
        twenty = " ".join(["word"] * 20)
        assert complexity_score(fifteen) == pytest.approx(complexity_score(twenty))

    def test_location_words_raise_the_score(self):
        plain = "The committee published its findings."
        geographic = "The committee published its findings in the harbor district."
        assert complexity_score(geographic) > complexity_score(plain)


class TestAdaptivePoolSize:
    def setup_method(self):
        self.config = SearchConfig(
            embedding_candidates=20, min_candidates=15, max_candidates=100
        )

    def test_generic_line_gets_wider_pool(self):
        generic = pool_size_for("It works.", self.config)
        specific = pool_size_for(
            "In late March the Northern Valley Transit Authority published its findings.",
            self.config,
        )
        assert generic > specific

    def test_pool_stays_within_bounds(self):
        for text in ["", "It works.", " ".join(["very"] * 50)]:
            size = pool_size_for(text, self.config)
            assert self.config.min_candidates <= size <= self.config.max_candidates

    def test_hard_max_is_respected_even_at_zero_complexity(self):
        config = SearchConfig(
            embedding_candidates=100, complexity_scaling_factor=5.0,
            min_candidates=15, max_candidates=60,
        )
        assert pool_size_for("", config) == 60


class TestDiversify:
    def setup_method(self):
        # Index -> source. Deliberately lopsided: source "a" owns 8 of 10.
        self.sources = ["a", "a", "a", "a", "a", "a", "a", "a", "b", "b"]

    def test_caps_how_many_come_from_one_source(self):
        # Sources: a x8, b x2. Taking in score order and capping at 2 gives
        # 0,1 (a is full), then 8,9 (b) -- and that is all that survives.
        scored = [(i, 1.0 - i * 0.01) for i in range(10)]  # sorted desc
        out = diversify(scored, lambda i: self.sources[i], k=5, max_per_source=2)
        assert [i for i, _ in out] == [0, 1, 8, 9]

    def test_preserves_score_order(self):
        scored = [(i, 1.0 - i * 0.01) for i in range(10)]
        out = diversify(scored, lambda i: self.sources[i], k=4, max_per_source=2)
        scores = [s for _, s in out]
        assert scores == sorted(scores, reverse=True)

    def test_zero_cap_disables_the_filter(self):
        scored = [(i, 1.0 - i * 0.01) for i in range(10)]
        out = diversify(scored, lambda i: self.sources[i], k=3, max_per_source=0)
        assert len(out) == 3
        assert [i for i, _ in out] == [0, 1, 2]

    def test_returns_fewer_than_k_when_caps_bind(self):
        # Every clip is from one source, so the cap is all that survives.
        sources = ["only"] * 5
        scored = [(i, 1.0 - i * 0.01) for i in range(5)]
        out = diversify(scored, lambda i: sources[i], k=5, max_per_source=2)
        # Correct behaviour: a short result beats padding with duplicates.
        assert len(out) == 2

    def test_under_fetch_is_what_this_prevents(self):
        # Balanced corpus: a x5, b x5. Plain top-3 is all one source.
        sources = ["a"] * 5 + ["b"] * 5
        scored = [(i, 1.0 - i * 0.01) for i in range(10)]

        naive = scored[:3]
        assert len({sources[i] for i, _ in naive}) == 1

        # With the 2x pre-fetch, capping after the fact actually has candidates
        # from a second source available to reach for.
        fetched = scored[:6]  # k * pre_fetch_multiplier
        fixed = diversify(fetched, lambda i: sources[i], k=3, max_per_source=2)
        assert len(fixed) == 3
        assert len({sources[i] for i, _ in fixed}) == 2

    def test_lopsided_corpus_cannot_be_diversified_by_a_2x_prefetch(self):
        # Worth stating rather than leaving as a surprise: pre-fetching k*2 is a
        # heuristic, not a guarantee. If one source owns almost the whole
        # top-scoring region, the fetched window contains nothing else to reach
        # for, and the cap can only shrink the result.
        sources = ["a"] * 8 + ["b"] * 2
        scored = [(i, 1.0 - i * 0.01) for i in range(10)]
        fetched = scored[:6]  # all six are source a
        assert len({sources[i] for i, _ in fetched}) == 1

        out = diversify(fetched, lambda i: sources[i], k=3, max_per_source=2)
        assert len(out) == 2  # the cap, not k


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self):
        v = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        scores = cosine_similarity(np.array([1.0, 0.0], dtype=np.float32), v)
        assert scores[0] == pytest.approx(1.0, abs=1e-5)
        assert scores[1] == pytest.approx(0.0, abs=1e-5)

    def test_opposite_vectors_score_minus_one(self):
        v = np.array([[-1.0, 0.0]], dtype=np.float32)
        scores = cosine_similarity(np.array([1.0, 0.0], dtype=np.float32), v)
        assert scores[0] == pytest.approx(-1.0, abs=1e-5)

    def test_empty_matrix_is_safe(self):
        scores = cosine_similarity(
            np.array([1.0, 0.0], dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
        )
        assert scores.shape == (0,)

    def test_zero_vector_does_not_divide_by_zero(self):
        v = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        scores = cosine_similarity(np.array([1.0, 0.0], dtype=np.float32), v)
        assert np.all(np.isfinite(scores))