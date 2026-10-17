"""Tests for SRT parsing and segment shaping."""

import pytest

from matching.srt import load_srt, merge_short_segments, parse_plain_text, parse_srt

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

# --- Plain pasted text -----------------------------------------------------
# The app offers a "Paste text" option, so pasting ordinary prose has to match
# rather than error. These pin that, and pin that it does NOT change how a real
# .srt is parsed.


def test_plain_prose_is_matched_not_rejected():
    text = "A sunrise over a busy city street.\nNobody speaks in this scene."
    segments = load_srt(text)
    assert len(segments) == 2
    assert segments[0].text == "A sunrise over a busy city street."
    assert segments[1].text == "Nobody speaks in this scene."


def test_plain_text_segments_do_not_overlap_or_merge():
    # Each pasted line is its own sentence and must stay its own shot. If
    # PLAIN_LINE_SECONDS ever dropped below MIN_SEGMENT_SECONDS, merge_short_
    # segments would fold neighbours together and one sentence would be matched
    # against one shot.
    segments = load_srt("One line here.\nTwo line here.\nThree line here.")
    assert len(segments) == 3
    for a, b in zip(segments, segments[1:]):
        assert a.end <= b.start


def test_plain_text_timeline_is_monotonic():
    segments = load_srt("Alpha.\nBeta.\nGamma.")
    assert [s.index for s in segments] == [0, 1, 2]
    assert segments == sorted(segments, key=lambda s: s.start)


def test_blank_lines_are_ignored_not_matched():
    segments = load_srt("Real line.\n\n\n   \nAnother real line.")
    assert len(segments) == 2


def test_real_srt_still_parses_exactly_as_before():
    # The fallback only fires when parse_srt finds nothing, so a real file is
    # untouched by it.
    assert load_srt(BASIC) == merge_short_segments(parse_srt(BASIC))


def test_srt_with_timing_is_never_treated_as_plain_text():
    # If this were falling through to parse_plain_text, the timing lines and the
    # index would each become a matched sentence. It doesn't: the short cue
    # merges forward into its neighbour exactly as a real .srt does.
    text = "1\n00:00:00,000 --> 00:00:01,000\nShort one.\n\n2\n00:00:01,000 --> 00:00:06,000\nLong one.\n"
    segments = load_srt(text)
    assert len(segments) == 1
    assert segments[0].text == "Short one. Long one."
    assert segments[0].start == 0.0 and segments[0].end == 6.0


def test_single_pasted_line_with_no_newline():
    segments = load_srt("Just one line, no trailing newline")
    assert len(segments) == 1
    assert segments[0].text == "Just one line, no trailing newline"


def test_html_tags_are_stripped_from_plain_lines():
    segments = load_srt("<i>Italic line.</i>\n{\an8} Positioned line.")
    assert [s.text for s in segments] == ["Italic line.", "Positioned line."]


def test_plain_text_durations_are_marked_as_estimated():
    # Printing a guessed "3.0s" as though it were measured would be a fabricated
    # number, so pasted lines render with a "~" and real .srt cues do not.
    pasted = load_srt("One line.\nTwo lines.")
    assert all(s.estimated_timing for s in pasted)
    assert all(s.duration_label.startswith("~") for s in pasted)

    real = load_srt(BASIC)
    assert not any(s.estimated_timing for s in real)
    assert all(not s.duration_label.startswith("~") for s in real)


def test_estimated_flag_survives_merging():
    # PLAIN_LINE_SECONDS >= MIN_SEGMENT_SECONDS means pasted lines never merge,
    # but if that ever changed the flag must not be dropped on the way through.
    segments = parse_plain_text("A.\nB.\nC.", seconds_per_line=1.0)
    assert all(s.estimated_timing for s in segments)
    merged = merge_short_segments(segments, min_seconds=3.0)
    assert merged and all(s.estimated_timing for s in merged)
