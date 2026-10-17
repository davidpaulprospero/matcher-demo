"""Tests for SRT parsing and segment shaping."""

import pytest

from matching.srt import load_srt, merge_short_segments, parse_srt

BASIC = """1
00:00:00,000 --> 00:00:04,000
The first line of the voiceover.

2
00:00:04,000 --> 00:00:08,500
The second line runs a little longer here.

3
00:00:08,500 --> 00:00:09,000
And.

4
00:00:09,000 --> 00:00:14,000
This cue is long enough to survive on its own.
"""


def test_parses_index_timing_and_text():
    segments = parse_srt(BASIC)
    assert len(segments) == 4
    assert segments[0].text == "The first line of the voiceover."
    assert segments[0].start == 0.0
    assert segments[0].end == 4.0


def test_dot_and_comma_millis_both_parse():
    segments = parse_srt("1\n00:00:01.500 --> 00:00:02.750\nDot form.\n")
    assert segments[0].start == 1.5
    assert segments[0].end == 2.75


def test_strips_html_and_ass_tags():
    segments = parse_srt("1\n00:00:00,000 --> 00:00:03,000\n<i>Styled</i> {\\an8} text.\n")
    assert segments[0].text == "Styled text."


def test_handles_windows_line_endings():
    segments = parse_srt(BASIC.replace("\n", "\r\n"))
    assert len(segments) == 4


def test_multi_line_text_is_joined():
    segments = parse_srt("1\n00:00:00,000 --> 00:00:04,000\nWrapped\nacross lines.\n")
    assert segments[0].text == "Wrapped across lines."


def test_missing_index_line_still_parses():
    segments = parse_srt("00:00:00,000 --> 00:00:04,000\nNo index here.\n")
    assert len(segments) == 1


def test_garbage_block_is_skipped_not_fatal():
    segments = parse_srt("not a subtitle block\n\n" + BASIC)
    assert len(segments) == 4


def test_short_segment_merges_forward():
    segments = parse_srt(BASIC)
    merged = merge_short_segments(segments, min_seconds=3.0)
    # The 0.5s "And." cue folds forward into the cue after it, which is why the
    # result has three segments rather than four.
    assert len(merged) == 3
    assert "And." in merged[2].text
    assert merged[2].text.endswith("survive on its own.")


def test_merge_preserves_start_time_of_kept_segment():
    segments = parse_srt(BASIC)
    merged = merge_short_segments(segments, min_seconds=3.0)
    # Merging goes forward so segment start times do not shift.
    assert merged[0].start == segments[0].start
    assert merged[1].start == segments[1].start


def test_last_short_segment_is_never_dropped():
    text = "1\n00:00:00,000 --> 00:00:05,000\nKeep me.\n\n2\n00:00:05,000 --> 00:00:05,400\nAnd me.\n"
    merged = merge_short_segments(parse_srt(text), min_seconds=3.0)
    assert len(merged) == 2
    assert "And me." in merged[-1].text


def test_empty_input():
    assert parse_srt("") == []
    assert merge_short_segments([]) == []


def test_indices_are_renumbered_after_merge():
    merged = merge_short_segments(parse_srt(BASIC), min_seconds=3.0)
    assert [s.index for s in merged] == list(range(len(merged)))


def test_load_srt_combines_both():
    merged = load_srt(BASIC)
    assert len(merged) == 3