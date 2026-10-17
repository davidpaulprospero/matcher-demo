"""The matching pipeline: voiceover in, ranked edit decisions out.

Order matters here and the order is the design:

    for each segment, in script order:
        1. complexity-scaled embedding search      (stateless, pool_size varies)
        2. drop sources over the variety window   (needs the timeline so far)
        3. apply reuse penalties                   (needs every prior choice)
        4. take the best surviving clip, record it

Every later stage depends on every earlier placement, which is exactly why the
stages cannot be reordered or run in parallel — and why a stage-boundary
checkpoint is the only thing that makes a failure recoverable. Re-running from
segment 1 would produce a different edit, not merely re-do work.

The result is a decision list, not a timeline file. Exporting OTIO/EDL needs the
real footage to exist on disk, which a synthetic corpus cannot provide; see
README.md for what is deliberately out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .embeddings import embed
from .search import Candidate, EmbeddingSearch, SearchConfig, complexity_score
from .reuse import ReuseTracker, TimelineVarietyTracker
from .srt import Segment


@dataclass
class Placement:
    """One segment's decision, with enough context to explain it."""

    segment: Segment
    clip: Optional[Dict]
    score: float
    base_score: float
    pool_size: int
    complexity: float
    # Candidates removed by the variety window, before reuse penalties.
    blocked_by_variety: int
    # True when the variety rule could not be satisfied and was applied as a
    # penalty instead. Surfaced in the UI because it means the edit is
    # repetitive by necessity, not by choice.
    variety_degraded: bool = False
    runner_up: Optional[Dict] = None

    @property
    def matched(self) -> bool:
        return self.clip is not None


@dataclass
class PipelineConfig:
    search: SearchConfig = field(default_factory=SearchConfig)
    max_reuse: int = 2
    reuse_penalty: float = 0.2
    max_source_file_reuse: int = 6
    source_file_penalty: float = 0.1
    timeline_window: float = 600.0
    max_repeats_per_window: int = 1
    # Applied instead of excluding, when excluding would empty the pool. Sized
    # to sit below a good match (~0.5) so violating variety costs real score
    # without making the alternative unusable.
    variety_fallback_penalty: float = 0.15
    # Below this, the UI flags the match as weak rather than hiding it.
    low_score_threshold: float = 0.35


class Matcher:
    def __init__(self, search: EmbeddingSearch, config: Optional[PipelineConfig] = None):
        self.search = search
        self.config = config or PipelineConfig()

    def run(self, segments: List[Segment], track: str = "V1") -> List[Placement]:
        """Match every segment. Segments are processed in script order."""
        if not segments:
            return []

        reuse = ReuseTracker(
            max_reuse=self.config.max_reuse,
            reuse_penalty=self.config.reuse_penalty,
            max_source_file_reuse=self.config.max_source_file_reuse,
            source_file_penalty=self.config.source_file_penalty,
        )
        variety = TimelineVarietyTracker(
            timeline_window=self.config.timeline_window,
            max_repeats=self.config.max_repeats_per_window,
        )

        # One embedding call for the whole script. Embedding line-by-line inside
        # the loop would reload nothing but would pay the per-call overhead
        # hundreds of times.
        vectors = embed([s.text for s in segments])

        placements: List[Placement] = []

        for segment, vector in zip(segments, vectors):
            candidates = self.search.search(vector, segment.text)
            position = segment.start

            blocked = variety.excluded_sources(track, position)
            allowed = [c for c in candidates if c.clip["source"] not in blocked]
            blocked_count = len(candidates) - len(allowed)

            # A hard variety block can empty the pool. It does, whenever the
            # script is shorter than the window and the corpus has fewer
            # sources than the window can consume: every line sits inside the
            # same 10 minutes, so after each source is used once there is
            # nothing left that is allowed. Refusing to place anything there is
            # correct in isolation and wrong in practice — a hole in the edit is
            # worse than a repeated shot.
            #
            # So the block degrades to a penalty instead of an exclusion, and
            # the segment is marked as having taken that fallback. A rule that
            # cannot be satisfied becomes a preference rather than a veto.
            degraded = not allowed and bool(candidates)
            if degraded:
                allowed = list(candidates)

            for candidate in allowed:
                clip_id = _clip_id(candidate)
                penalty = reuse.penalty(clip_id, candidate.clip["source"])
                if degraded and candidate.clip["source"] in blocked:
                    # Scale with how badly the rule is being violated, so a
                    # mildly-overused source still loses to a fresh one.
                    penalty += self.config.variety_fallback_penalty
                candidate.penalty_applied = penalty
                candidate.similarity = max(0.0, candidate.similarity - penalty)

            # Re-filter after penalties: reuse can drop a clip below others it
            # was out-ranked by, so sorting the pre-penalty order would keep a
            # worse pick.
            allowed = [c for c in allowed if reuse.can_use(_clip_id(c), c.clip["source"])]
            allowed.sort(key=lambda c: c.similarity, reverse=True)

            if allowed:
                best = allowed[0]
                clip_id = _clip_id(best)
                reuse.record_usage(clip_id, best.clip["source"])
                variety.record_usage(track, best.clip["source"], position)

                placements.append(Placement(
                    segment=segment,
                    clip=best.clip,
                    score=best.similarity,
                    base_score=best.base_similarity,
                    pool_size=best.pool_size,
                    complexity=complexity_score(segment.text),
                    blocked_by_variety=blocked_count,
                    variety_degraded=degraded,
                    runner_up=allowed[1].clip if len(allowed) > 1 else None,
                ))
            else:
                # No clip survives even after the fallback. Recorded rather than
                # skipped — a hole in the edit is information, and hiding it
                # would make the timeline look complete when it isn't.
                placements.append(Placement(
                    segment=segment,
                    clip=None,
                    score=0.0,
                    base_score=0.0,
                    pool_size=len(candidates),
                    complexity=complexity_score(segment.text),
                    blocked_by_variety=blocked_count,
                    variety_degraded=degraded,
                ))

        return placements

    def variety_stats(self, placements: List[Placement], track: str = "V1") -> Dict[str, object]:
        """Re-derive variety stats from the placements, for reporting."""
        tracker = TimelineVarietyTracker(
            timeline_window=self.config.timeline_window,
            max_repeats=self.config.max_repeats_per_window,
        )
        for p in placements:
            if p.clip:
                tracker.record_usage(track, p.clip["source"], p.segment.start)
        return tracker.stats(track)


def _clip_id(candidate: Candidate) -> str:
    return ReuseTracker.clip_id(
        candidate.clip["source"],
        candidate.clip["start_time"],
        candidate.clip["start_time"] + candidate.clip["duration"],
    )