"""Reuse and timeline-variety tracking.

These two trackers solve one problem: the best-scoring clip is usually the best
-scoring clip for the next three lines too. Left alone, a 20-line voiceover
gets the same five shots repeated, and the edit looks worse than random.

They work at different granularities on purpose:

- `ReuseTracker` is a running tally. It penalises a clip in proportion to how
  many times it has been used so far. It never hard-blocks on clip-level reuse
  (default `max_reuse=2`) because a hard block with no fallback means the
  segment gets no clip at all — an empty slot is worse than a repeat.

- `TimelineVarietyTracker` is local to a time window. It hard-blocks a *source
  video* appearing more than `max_repeats` times within the last
  `timeline_window` seconds. Blocking is safe at this level because the pool is
  large enough to still contain something else from a different shoot.

A hard block that can leave a hole and a soft penalty that can't — that split is
the decision worth reading the code for.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from typing import Dict, List, Set, Tuple


class ReuseTracker:
    """Running count of clip and source-video use, with confidence penalties."""

    def __init__(
        self,
        max_reuse: int = 2,
        reuse_penalty: float = 0.2,
        max_source_file_reuse: int = 0,  # 0 = unlimited
        source_file_penalty: float = 0.1,
    ):
        self.max_reuse = max_reuse
        self.reuse_penalty = reuse_penalty
        self.max_source_file_reuse = max_source_file_reuse
        self.source_file_penalty = source_file_penalty

        self.usage_count: Dict[str, int] = {}
        self.source_file_count: Dict[str, int] = {}

    @staticmethod
    def clip_id(source: str, start: float, end: float) -> str:
        """Identity of a clip: same source *and* same time range."""
        return f"{source}:{start:.2f}-{end:.2f}"

    def get_usage_count(self, clip_id: str) -> int:
        return self.usage_count.get(clip_id, 0)

    def get_source_usage(self, source: str) -> int:
        return self.source_file_count.get(source, 0)

    def record_usage(self, clip_id: str, source: str) -> None:
        self.usage_count[clip_id] = self.usage_count.get(clip_id, 0) + 1
        self.source_file_count[source] = self.source_file_count.get(source, 0) + 1

    def can_use(self, clip_id: str, source: str) -> bool:
        """False once a clip is over its own reuse limit.

        Returns False rather than clamping the penalty, because a clip used to
        exhaustion should stop being a candidate at all. The caller keeps the
        rest of the pool.
        """
        if self.get_usage_count(clip_id) >= self.max_reuse:
            return False
        if self.max_source_file_reuse > 0:
            if self.get_source_usage(source) >= self.max_source_file_reuse:
                return False
        return True

    def penalty(self, clip_id: str, source: str) -> float:
        """Confidence to subtract, from clip reuse plus source-file overuse.

        The source penalty is *escalating*: flat cost up to half the limit,
        then it climbs. Source overuse is the failure that looks worst on
        screen — nine shots from one shoot in a row reads as a mistake — so it
        is worth penalising harder than an individual clip repeating.
        """
        clip_penalty = self.get_usage_count(clip_id) * self.reuse_penalty

        source_penalty = 0.0
        if self.max_source_file_reuse > 0:
            count = self.get_source_usage(source)
            threshold = self.max_source_file_reuse // 2
            if count > threshold:
                source_penalty = (count - threshold) * self.source_file_penalty

        return clip_penalty + source_penalty

    def adjust(self, clip_id: str, source: str, similarity: float) -> float:
        """Apply the penalty, floored at 0 so a bad score can't go negative."""
        return max(0.0, similarity - self.penalty(clip_id, source))


class TimelineVarietyTracker:
    """Hard-blocks a source video repeating inside a sliding time window."""

    def __init__(self, timeline_window: float = 600.0, max_repeats: int = 1):
        self.timeline_window = timeline_window
        self.max_repeats = max_repeats

        # {track: [(position, source), ...]} kept sorted by position, so the
        # window start is a bisect rather than a scan.
        self.track_usage: Dict[str, List[Tuple[float, str]]] = defaultdict(list)

    def excluded_sources(self, track: str, position: float) -> Set[str]:
        """Sources this track has already used too often before `position`."""
        usages = self.track_usage.get(track)
        if not usages:
            return set()

        window_start = position - self.timeline_window
        # First usage at or after window_start. Everything before that has
        # fallen out of the window and cannot count against us.
        start_idx = bisect_left([pos for pos, _ in usages], window_start)

        counts: Dict[str, int] = defaultdict(int)
        for pos, source in usages[start_idx:]:
            if pos >= position:
                break
            counts[source] += 1

        return {src for src, count in counts.items() if count >= self.max_repeats}

    def record_usage(self, track: str, source: str, position: float) -> None:
        """Record a placement. Maintains sort order so the bisect stays valid."""
        usages = self.track_usage[track]
        entry = (position, source)

        if not usages or usages[-1][0] <= position:
            usages.append(entry)  # the common case: edits run in order
        else:
            usages.insert(bisect_left([pos for pos, _ in usages], position), entry)

    def stats(self, track: str) -> Dict[str, object]:
        """Longest run of a single source, as a variety check.

        A timeline where the worst source run is 3+ shots looks repetitive even
        though no rule was violated, which is why this is worth surfacing.
        """
        usages = self.track_usage.get(track, [])
        if not usages:
            return {"total_clips": 0, "unique_sources": 0, "longest_run": 0, "longest_run_source": None}

        counts: Dict[str, int] = defaultdict(int)
        longest = 0
        longest_src = None
        run_src = None
        run = 0

        for _, source in usages:
            counts[source] += 1
            run = run + 1 if source == run_src else 1
            run_src = source
            if run > longest:
                longest = run
                longest_src = source

        return {
            "total_clips": len(usages),
            "unique_sources": len(counts),
            "longest_run": longest,
            "longest_run_source": longest_src,
        }