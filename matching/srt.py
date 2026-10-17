"""SRT parsing and segment shaping.

A voiceover script arrives as .srt. Two things about that file are awkward and
both are handled here rather than downstream:

1. **Segments are not sentences.** A subtitle is whatever fit on screen, so a
   30-second sentence is routinely split into six two-second cues. Matching
   each cue separately picks six shots for one sentence.

2. **Cues are hard-limited by display width.** So you get fragments like
   "and then the" as their own segment, which is not matchable to anything.

`parse_srt` reads the file; `merge_short_segments` fixes the shape.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

# 00:00:12,500 --> 00:00:14,000   (comma or dot, 2-3 digit hours)
_TIMING = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)
_INDEX = re.compile(r"^\d+$")

# Anything shorter than this merges into its neighbour. Subtitle cues are
# usually ~2s; a line you could read in one breath is ~3s.
MIN_SEGMENT_SECONDS = 3.0

# Tags a subtitling tool leaves behind (<i>, {\an8}, ASS overrides).
_TAGS = re.compile(r"<[^>]+>|\{[^}]*\}")


@dataclass
class Segment:
    """One matchable unit of voiceover."""

    index: int
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"Segment({self.index}, {self.start:.1f}-{self.end:.1f}, {self.text!r})"


def _to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


def parse_srt(content: str) -> List[Segment]:
    """Parse .srt text into segments.

    Tolerant of the things real .srt files do: a missing index line, \r\n
    line endings, HTML tags, and text spanning several lines.
    """
    segments: List[Segment] = []

    # Normalise newlines first so the block split behaves on Windows exports.
    blocks = re.split(r"\n\s*\n", content.replace("\r\n", "\n").replace("\r", "\n").strip())

    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue

        # Drop a leading sequence number if present.
        if _INDEX.match(lines[0].strip()):
            lines = lines[1:]

        if not lines:
            continue

        match = _TIMING.search(lines[0])
        if not match:
            continue

        g = match.groups()
        start = _to_seconds(*g[0:4])
        end = _to_seconds(*g[4:8])

        text = " ".join(lines[1:]) if len(lines) > 1 else ""
        text = _TAGS.sub("", text).strip()
        text = re.sub(r"\s+", " ", text)

        if text:
            segments.append(Segment(len(segments), start, end, text))

    return segments


def merge_short_segments(segments: List[Segment], min_seconds: float = MIN_SEGMENT_SECONDS) -> List[Segment]:
    """Fold sub-threshold cues into the next one.

    Merging always goes *forward*: a short cue joins the segment that follows,
    rather than pulling the previous one forward. That keeps the timeline
    start-time of segment N stable, which matters because the matched clip is
    placed at that time.

    The final segment is never merged away even if it is short — dropping the
    last line of a script loses the ending.
    """
    if not segments:
        return []

    merged: List[Segment] = []
    pending: Segment | None = None

    for seg in segments:
        if pending is not None:
            seg = Segment(pending.index, pending.start, seg.end,
                          f"{pending.text} {seg.text}".strip())

        is_last = seg.index == segments[-1].index
        if seg.duration < min_seconds and not is_last:
            pending = seg
            continue

        merged.append(seg)
        pending = None

    if pending is not None:
        merged.append(pending)

    return [Segment(i, s.start, s.end, s.text) for i, s in enumerate(merged)]


def load_srt(content: str, min_seconds: float = MIN_SEGMENT_SECONDS) -> List[Segment]:
    """Parse and shape in one call. This is what the app uses."""
    return merge_short_segments(parse_srt(content), min_seconds=min_seconds)