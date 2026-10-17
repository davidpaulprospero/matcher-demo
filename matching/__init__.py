"""Matching demo package."""

from .srt import Segment, load_srt, merge_short_segments, parse_srt
from .search import Candidate, EmbeddingSearch, SearchConfig, complexity_score, pool_size_for
from .reuse import ReuseTracker, TimelineVarietyTracker
from .pipeline import Matcher, PipelineConfig, Placement

__all__ = [
    "Segment",
    "load_srt",
    "merge_short_segments",
    "parse_srt",
    "Candidate",
    "EmbeddingSearch",
    "SearchConfig",
    "complexity_score",
    "pool_size_for",
    "ReuseTracker",
    "TimelineVarietyTracker",
    "Matcher",
    "PipelineConfig",
    "Placement",
]