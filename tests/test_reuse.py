"""Tests for reuse penalties and windowed variety blocking."""

import pytest

from matching.reuse import ReuseTracker, TimelineVarietyTracker


class TestReuseTracker:
    def test_fresh_clip_has_no_penalty(self):
        tracker = ReuseTracker()
        assert tracker.penalty("clip-1", "shoot-a") == 0.0

    def test_clip_penalty_grows_with_use(self):
        tracker = ReuseTracker(reuse_penalty=0.2)
        tracker.record_usage("clip-1", "shoot-a")
        assert tracker.penalty("clip-1", "shoot-a") == pytest.approx(0.2)
        tracker.record_usage("clip-1", "shoot-a")
        assert tracker.penalty("clip-1", "shoot-a") == pytest.approx(0.4)

    def test_clip_over_limit_becomes_unusable(self):
        tracker = ReuseTracker(max_reuse=2)
        assert tracker.can_use("clip-1", "shoot-a")
        tracker.record_usage("clip-1", "shoot-a")
        assert tracker.can_use("clip-1", "shoot-a")
        tracker.record_usage("clip-1", "shoot-a")
        assert not tracker.can_use("clip-1", "shoot-a")

    def test_adjust_floors_at_zero(self):
        tracker = ReuseTracker(reuse_penalty=5.0)
        tracker.record_usage("clip-1", "shoot-a")
        assert tracker.adjust("clip-1", "shoot-a", 0.3) == 0.0

    def test_source_penalty_escalates_past_half_the_limit(self):
        tracker = ReuseTracker(max_source_file_reuse=6, source_file_penalty=0.1)
        # Usage must be in-place before the penalty applies.
        for i in range(3):
            tracker.record_usage(f"clip-{i}", "shoot-a")
        assert tracker.penalty("clip-new", "shoot-a") == 0.0  # at threshold

        tracker.record_usage("clip-3", "shoot-a")  # count 4 > 3
        assert tracker.penalty("clip-new", "shoot-a") == pytest.approx(0.1)

        tracker.record_usage("clip-4", "shoot-a")  # count 5
        assert tracker.penalty("clip-new", "shoot-a") == pytest.approx(0.2)

    def test_source_limit_blocks_when_enabled(self):
        tracker = ReuseTracker(max_source_file_reuse=2)
        assert tracker.can_use("c1", "shoot-a")
        tracker.record_usage("c1", "shoot-a")
        tracker.record_usage("c2", "shoot-a")
        assert not tracker.can_use("c3", "shoot-a")

    def test_source_limit_ignored_when_zero(self):
        tracker = ReuseTracker(max_source_file_reuse=0)
        for i in range(20):
            tracker.record_usage(f"clip-{i}", "shoot-a")
        assert tracker.can_use("fresh", "shoot-a")

    def test_clip_ids_differ_by_time_range(self):
        a = ReuseTracker.clip_id("shoot-a", 0.0, 5.0)
        b = ReuseTracker.clip_id("shoot-a", 5.0, 10.0)
        assert a != b


class TestTimelineVarietyTracker:
    def test_nothing_excluded_at_the_start(self):
        tracker = TimelineVarietyTracker()
        assert tracker.excluded_sources("V1", 0.0) == set()

    def test_source_blocked_after_max_repeats(self):
        tracker = TimelineVarietyTracker(max_repeats=1)
        tracker.record_usage("V1", "shoot-a", 100.0)
        assert tracker.excluded_sources("V1", 110.0) == {"shoot-a"}

    def test_block_lifts_once_the_window_passes(self):
        tracker = TimelineVarietyTracker(timeline_window=600.0, max_repeats=1)
        tracker.record_usage("V1", "shoot-a", 100.0)
        # 901s later the 600s window no longer contains the earlier placement.
        assert tracker.excluded_sources("V1", 1000.0) == set()

    def test_tracks_are_independent(self):
        tracker = TimelineVarietyTracker(max_repeats=1)
        tracker.record_usage("V1", "shoot-a", 100.0)
        assert tracker.excluded_sources("V2", 110.0) == set()

    def test_future_placements_do_not_count(self):
        tracker = TimelineVarietyTracker(max_repeats=1)
        tracker.record_usage("V1", "shoot-a", 500.0)
        # Querying at 100 must not see the placement that happens at 500.
        assert tracker.excluded_sources("V1", 100.0) == set()

    def test_window_boundary_is_inclusive_at_both_ends(self):
        # The window is [position - timeline_window, position), so a placement
        # exactly at the window start still counts against us and one exactly
        # at the current position does not.
        tracker = TimelineVarietyTracker(timeline_window=600.0, max_repeats=1)
        tracker.record_usage("V1", "shoot-a", 100.0)
        assert tracker.excluded_sources("V1", 699.0) == {"shoot-a"}
        assert tracker.excluded_sources("V1", 700.0) == {"shoot-a"}
        assert tracker.excluded_sources("V1", 701.0) == set()

    def test_queried_position_itself_does_not_count(self):
        tracker = TimelineVarietyTracker(timeline_window=600.0, max_repeats=1)
        tracker.record_usage("V1", "shoot-a", 400.0)
        # Placing at 400 then asking about 400 is a first placement, not a repeat.
        assert tracker.excluded_sources("V1", 400.0) == set()

    def test_out_of_order_insert_keeps_list_sorted(self):
        tracker = TimelineVarietyTracker()
        tracker.record_usage("V1", "a", 300.0)
        tracker.record_usage("V1", "b", 100.0)  # arrives late
        positions = [pos for pos, _ in tracker.track_usage["V1"]]
        assert positions == sorted(positions)

    def test_max_repeats_two_allows_one_repeat(self):
        tracker = TimelineVarietyTracker(max_repeats=2)
        tracker.record_usage("V1", "shoot-a", 100.0)
        assert tracker.excluded_sources("V1", 110.0) == set()
        tracker.record_usage("V1", "shoot-a", 120.0)
        assert tracker.excluded_sources("V1", 130.0) == {"shoot-a"}


class TestVarietyStats:
    def test_empty_track(self):
        tracker = TimelineVarietyTracker()
        assert tracker.stats("V1")["total_clips"] == 0

    def test_longest_run_is_reported(self):
        tracker = TimelineVarietyTracker()
        for i, source in enumerate(["a", "a", "a", "b", "a"]):
            tracker.record_usage("V1", source, float(i))
        stats = tracker.stats("V1")
        assert stats["total_clips"] == 5
        assert stats["unique_sources"] == 2
        assert stats["longest_run"] == 3
        assert stats["longest_run_source"] == "a"