"""Tests for the pipeline's stage ordering and its fallback behaviour.

The searches here use a fake embedder, so the whole pipeline runs without
loading a model. That keeps the test suite instant and, more usefully, lets
these tests assert on *ordering* — which stage saw what — rather than on
whatever a model happens to score today.
"""

import numpy as np
import pytest

import matching.embeddings as embeddings_mod
from matching import (
    EmbeddingSearch,
    Matcher,
    PipelineConfig,
    SearchConfig,
    load_srt,
)
from matching.srt import Segment

EMBED_DIM = 8


@pytest.fixture(autouse=True)
def fake_embedder(monkeypatch):
    """Replace the model with a deterministic keyword-based stand-in.

    Embeds by hashing words onto the unit sphere, so lines sharing a word land
    near each other. Not semantic — but stable and interpretable, which is what
    ordering tests need.
    """
    def _embed(texts):
        vectors = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                vectors[i, hash(word) % EMBED_DIM] += 1.0
            vectors[i, 0] += 0.1  # never all-zero
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    monkeypatch.setattr("matching.pipeline.embed", _embed)
    return _embed


def make_catalogue(sources=2, per_source=4):
    """A tiny catalogue with predictable captions."""
    clips = []
    for s in range(sources):
        for i in range(per_source):
            clips.append({
                "id": f"s{s}-c{i}",
                "source": f"source-{s}",
                "caption": f"source {s} clip {i} {'ocean' * (i + 1)}",
                "camera": "static wide",
                "has_face": False,
                "place": "ocean",
                "mood": "calm",
                "start_time": float(i * 5),
                "duration": 4.0,
            })
    emb = np.random.RandomState(0).randn(len(clips), EMBED_DIM).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return clips, emb


def segments(*texts):
    return [
        Segment(i, float(i * 10), float(i * 10 + 8), text)
        for i, text in enumerate(texts)
    ]


class TestHappyPath:
    def test_every_segment_gets_a_clip(self):
        clips, emb = make_catalogue()
        search = EmbeddingSearch(clips, emb)
        placements = Matcher(search, PipelineConfig()).run(
            segments("ocean wide", "ocean calm", "ocean deep")
        )
        assert len(placements) == 3
        assert all(p.matched for p in placements)

    def test_no_segments_returns_empty(self):
        clips, emb = make_catalogue()
        placements = Matcher(EmbeddingSearch(clips, emb)).run([])
        assert placements == []

    def test_empty_catalogue_yields_no_matches_not_a_crash(self):
        search = EmbeddingSearch([], np.zeros((0, EMBED_DIM), dtype=np.float32))
        placements = Matcher(search).run(segments("ocean"))
        assert len(placements) == 1
        assert not placements[0].matched

    def test_score_never_exceeds_one_after_a_penalty(self):
        clips, emb = make_catalogue()
        search = EmbeddingSearch(clips, emb)
        config = PipelineConfig(max_reuse=5, reuse_penalty=0.5)
        placements = Matcher(search, config).run(
            segments("ocean", "ocean", "ocean", "ocean")
        )
        assert all(0.0 <= p.score <= 1.0 for p in placements)


class TestVarietyFallback:
    """The rule degrades to a penalty when obeying it would empty the pool."""

    def test_no_gaps_when_the_corpus_is_smaller_than_the_window(self):
        # One source, max_repeats=1: after the first line every candidate is
        # blocked. The fallback must place them anyway.
        clips, emb = make_catalogue(sources=1, per_source=4)
        search = EmbeddingSearch(clips, emb)
        config = PipelineConfig(max_repeats_per_window=1, timeline_window=600.0)
        placements = Matcher(search, config).run(
            segments("ocean", "ocean", "ocean", "ocean")
        )
        assert all(p.matched for p in placements)
        assert all(p.variety_degraded for p in placements[1:])

    def test_first_line_is_not_degraded(self):
        clips, emb = make_catalogue(sources=1, per_source=4)
        placements = Matcher(EmbeddingSearch(clips, emb), PipelineConfig()).run(
            segments("ocean", "ocean")
        )
        assert not placements[0].variety_degraded

    def test_fallback_is_flagged_in_blocked_count(self):
        clips, emb = make_catalogue(sources=1, per_source=4)
        placements = Matcher(EmbeddingSearch(clips, emb), PipelineConfig()).run(
            segments("ocean", "ocean")
        )
        assert placements[1].blocked_by_variety > 0

    def test_fallback_penalty_actually_lowers_the_score(self):
        # Same script, fallback penalty on vs off: the degraded scores must be
        # strictly lower, which is what makes the fallback a preference rather
        # than a silent override.
        clips, emb = make_catalogue(sources=1, per_source=4)
        lines = segments("ocean", "ocean", "ocean")

        soft = Matcher(
            EmbeddingSearch(clips, emb),
            PipelineConfig(variety_fallback_penalty=0.0),
        ).run(lines)
        hard = Matcher(
            EmbeddingSearch(clips, emb),
            PipelineConfig(variety_fallback_penalty=0.4),
        ).run(lines)

        assert hard[2].score < soft[2].score

    def test_fallback_still_honours_the_clip_reuse_limit(self):
        # max_reuse=1 means a given clip cannot repeat even under the fallback.
        # The corpus is one source x 4 clips over 6 lines, so it CAN fill.
        clips, emb = make_catalogue(sources=1, per_source=4)
        search = EmbeddingSearch(clips, emb)
        placements = Matcher(
            search, PipelineConfig(max_reuse=1, max_repeats_per_window=1)
        ).run(segments(*["ocean"] * 6))

        used = [p.clip["id"] for p in placements if p.matched]
        assert len(used) == len(set(used)), "a clip repeated despite max_reuse=1"

    def test_gap_is_reported_when_the_corpus_is_genuinely_exhausted(self):
        # One clip, max_reuse=1, three lines. Nothing can be placed on lines
        # 2 and 3, and that must be visible rather than silently padded.
        clips, emb = make_catalogue(sources=1, per_source=1)
        placements = Matcher(
            EmbeddingSearch(clips, emb), PipelineConfig(max_reuse=1)
        ).run(segments("ocean", "ocean", "ocean"))

        assert placements[0].matched
        assert not placements[1].matched
        assert not placements[2].matched

    def test_wide_window_never_degrades(self):
        # Each line far enough apart that nothing falls in the window.
        clips, emb = make_catalogue(sources=1, per_source=4)
        far_apart = [
            Segment(0, 0.0, 8.0, "ocean"),
            Segment(1, 5000.0, 5008.0, "ocean"),
            Segment(2, 10000.0, 10008.0, "ocean"),
        ]
        placements = Matcher(
            EmbeddingSearch(clips, emb), PipelineConfig(timeline_window=600.0)
        ).run(far_apart)
        assert not any(p.variety_degraded for p in placements)


class TestOrdering:
    def test_segments_are_processed_in_script_order(self):
        clips, emb = make_catalogue(sources=3, per_source=2)
        lines = segments("alpha", "beta", "gamma")
        placements = Matcher(EmbeddingSearch(clips, emb)).run(lines)
        assert [p.segment.index for p in placements] == [0, 1, 2]
        assert [p.segment.text for p in placements] == ["alpha", "beta", "gamma"]

    def test_pool_size_is_recorded_per_placement(self):
        clips, emb = make_catalogue()
        placements = Matcher(EmbeddingSearch(clips, emb)).run(segments("ocean"))
        assert placements[0].pool_size > 0

    def test_variety_stats_agree_with_the_placements(self):
        clips, emb = make_catalogue(sources=2, per_source=3)
        search = EmbeddingSearch(clips, emb)
        matcher = Matcher(search, PipelineConfig())
        placements = matcher.run(segments("ocean", "ocean", "ocean"))
        stats = matcher.variety_stats(placements)
        assert stats["total_clips"] == sum(1 for p in placements if p.matched)


class TestSampleScriptEndToEnd:
    def test_the_shipped_sample_fully_matches(self):
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        sample = root / "data" / "sample.srt"
        catalog_path = root / "data" / "catalog.json"
        if not sample.exists() or not catalog_path.exists():
            pytest.skip("sample data not generated")

        import json

        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        # Random stand-in vectors: this test is about the pipeline running
        # end-to-end and leaving no gaps, not about semantic quality.
        emb = np.random.RandomState(1).randn(
            len(catalog["clips"]), EMBED_DIM
        ).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)

        parsed = load_srt(sample.read_text(encoding="utf-8"))
        placements = Matcher(
            EmbeddingSearch(catalog["clips"], emb), PipelineConfig()
        ).run(parsed)

        assert len(placements) == len(parsed)
        assert all(p.matched for p in placements), "the shipped sample must leave no gaps"