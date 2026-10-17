"""Tests for the generated corpus.

These guard the data, not the code. A corpus regression is easy to miss because
nothing errors — the demo just gets subtly worse, and "subtly worse" is what
makes a matcher's output untrustworthy in the first place.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "data" / "catalog.json"
GENERATOR = ROOT / "scripts" / "generate_corpus.py"


@pytest.fixture(scope="module")
def catalog():
    if not CATALOG.exists():
        pytest.skip("corpus not generated; run scripts/generate_corpus.py")
    return json.loads(CATALOG.read_text(encoding="utf-8"))


class TestProvenance:
    def test_is_marked_synthetic(self, catalog):
        # The single most important assertion in this file. The UI's claim
        # depends on it, and matching/embeddings.py refuses to load a catalogue
        # without it.
        assert catalog["synthetic"] is True

    def test_carries_a_note_saying_where_it_came_from(self, catalog):
        assert "generated" in catalog["note"].lower()
        assert "not" in catalog["note"].lower()  # "not scraped from..."

    def test_records_the_seed(self, catalog):
        assert isinstance(catalog["seed"], int)

    def test_regeneration_is_reproducible(self):
        """Re-running the generator must produce byte-identical output.

        Without this the corpus is unreviewable: a reader could not tell
        whether a change to the data came from the code or from the run.
        """
        original = CATALOG.read_bytes()
        subprocess.run(
            [sys.executable, str(GENERATOR)],
            check=True, capture_output=True, cwd=str(ROOT),
        )
        assert CATALOG.read_bytes() == original


class TestUniqueness:
    def test_every_caption_is_distinct(self, catalog):
        captions = [c["caption"] for c in catalog["clips"]]
        duplicates = {c for c in captions if captions.count(c) > 1}
        assert not duplicates, (
            f"{len(duplicates)} duplicate captions: two different clips showing "
            "identical text makes a runner-up indistinguishable from the pick"
        )

    def test_every_clip_id_is_distinct(self, catalog):
        ids = [c["id"] for c in catalog["clips"]]
        assert len(set(ids)) == len(ids)

    def test_id_prefix_agrees_with_the_source(self, catalog):
        # The clip id is the reuse tracker's identity key. If the prefix and the
        # source field could disagree, two clips could look identical to it.
        for c in catalog["clips"]:
            assert c["id"].startswith(c["source"].split("/")[-1] + "-")


class TestShape:
    def test_clip_count_matches_the_recorded_field(self, catalog):
        assert catalog["clip_count"] == len(catalog["clips"])

    def test_enough_clips_to_make_matching_meaningful(self, catalog):
        assert len(catalog["clips"]) >= 150

    def test_multiple_sources(self, catalog):
        sources = {c["source"] for c in catalog["clips"]}
        # The source-diversity feature is untestable with one source.
        assert len(sources) >= 5

    def test_no_source_dominates_the_pool(self, catalog):
        # If one source owns most clips, the diversity cap has to fight the
        # corpus rather than the ranking.
        counts = {}
        for c in catalog["clips"]:
            counts[c["source"]] = counts.get(c["source"], 0) + 1
        assert max(counts.values()) / len(catalog["clips"]) < 0.2

    def test_durations_are_plausible(self, catalog):
        for c in catalog["clips"]:
            assert 2.0 <= c["duration"] <= 12.0

    def test_start_times_do_not_overlap_within_a_source(self, catalog):
        by_source = {}
        for c in catalog["clips"]:
            by_source.setdefault(c["source"], []).append(c)
        for source, clips in by_source.items():
            ordered = sorted(clips, key=lambda c: c["start_time"])
            for a, b in zip(ordered, ordered[1:]):
                assert a["start_time"] + a["duration"] <= b["start_time"] + 1e-6, (
                    f"{source}: {a['id']} overlaps {b['id']}"
                )

    def test_required_fields_present(self, catalog):
        required = {
            "id", "source", "caption", "camera", "lighting",
            "has_face", "place", "mood", "start_time", "duration",
        }
        for c in catalog["clips"][:5]:
            assert required <= set(c)

    def test_shoot_sizes_are_varied(self, catalog):
        # Variety in shoot size is what forces the pool-size and per-source
        # logic to do real work.
        counts = {}
        for c in catalog["clips"]:
            counts[c["source"]] = counts.get(c["source"], 0) + 1
        assert len(set(counts.values())) > 1


class TestEmbeddings:
    def test_embeddings_match_the_catalogue(self):
        emb_path = ROOT / "data" / "embeddings.json"
        if not emb_path.exists() or not CATALOG.exists():
            pytest.skip("embeddings not built")

        from matching.embeddings import MODEL_NAME, load_embeddings

        catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
        raw = json.loads(emb_path.read_text(encoding="utf-8"))
        vectors = load_embeddings(emb_path)

        assert vectors.shape[0] == len(catalog["clips"]), (
            "stale embeddings; re-run scripts/build_embeddings.py"
        )
        assert vectors.shape[1] == 384
        assert raw["model"] == MODEL_NAME

    def test_embeddings_are_unit_length(self):
        emb_path = ROOT / "data" / "embeddings.json"
        if not emb_path.exists():
            pytest.skip("embeddings not built")

        import numpy as np

        from matching.embeddings import load_embeddings

        norms = np.linalg.norm(load_embeddings(emb_path), axis=1)
        assert np.allclose(norms, 1.0, atol=1e-3), "embeddings must be L2-normalised"

    def test_embeddings_cover_every_clip_id(self):
        emb_path = ROOT / "data" / "embeddings.json"
        if not emb_path.exists() or not CATALOG.exists():
            pytest.skip("embeddings not built")

        catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
        raw = json.loads(emb_path.read_text(encoding="utf-8"))
        assert raw["clip_ids"] == [c["id"] for c in catalog["clips"]]