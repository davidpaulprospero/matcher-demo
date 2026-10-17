"""Measure peak memory of the running app.

Why this exists
---------------
Render's free instance has 512MB of RAM. If the demo needs more than that it
does not deploy, and the failure mode is bad: Render kills the container
mid-request, so the page loads and then dies, which reads as "the site is
broken" rather than "too big for the tier".

Better to find out here. Reports resident memory after a full cold load plus a
match run — the worst case, which is what the first real visitor hits.

Run:  python scripts/check_memory.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FREE_TIER_LIMIT_MB = 512


def rss_mb() -> float:
    """Resident set size in MB.

    Uses psutil if available. Falls back to tasklist on Windows and /proc on
    Linux — psutil is not a dependency of the app itself, only of this check.
    """
    try:
        import psutil  # type: ignore

        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except ImportError:
        pass

    if os.name == "nt":
        import subprocess

        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {os.getpid()}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True,
        ).stdout
        # "name","pid","session","#","5,123 K"
        for line in out.strip().splitlines():
            parts = [p.strip('" ') for p in line.split('","')]
            if len(parts) >= 5 and parts[-1].replace(",", "").strip().endswith("K"):
                return int(parts[-1].replace(",", "").strip()[:-1]) / 1024
        return 0.0

    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0.0


def main() -> None:
    baseline = rss_mb()
    print(f"interpreter only      {baseline:7.1f} MB")

    import numpy  # noqa: F401

    after_numpy = rss_mb()
    print(f"+ numpy               {after_numpy:7.1f} MB")

    from matching.embeddings import MODEL_NAME, embed, load_catalog, load_embeddings

    catalog = load_catalog(ROOT / "data" / "catalog.json")
    embeddings = load_embeddings(ROOT / "data" / "embeddings.json")
    print(f"+ catalogue loaded    {rss_mb():7.1f} MB  ({len(catalog['clips'])} clips)")

    # The real cost: loading MiniLM. This is what a cold start pays.
    vectors = embed([c["caption"] for c in catalog["clips"][:64]])
    after_model = rss_mb()
    print(f"+ {MODEL_NAME} loaded {after_model:7.1f} MB  (this is the jump that matters)")

    from matching import EmbeddingSearch, Matcher, PipelineConfig, SearchConfig, load_srt

    search = EmbeddingSearch(catalog["clips"], embeddings)
    segments = load_srt((ROOT / "data" / "sample.srt").read_text(encoding="utf-8"))
    placements = Matcher(
        search, PipelineConfig(search=SearchConfig(embedding_candidates=20))
    ).run(segments)
    peak = rss_mb()

    print(f"+ full match run      {peak:7.1f} MB  ({len(placements)} placements)")

    headroom = FREE_TIER_LIMIT_MB - peak
    verdict = "FITS" if headroom > 0 else "DOES NOT FIT"

    print()
    print(f"peak                  {peak:7.1f} MB")
    print(f"free tier limit       {FREE_TIER_LIMIT_MB:7.1f} MB")
    print(f"headroom              {headroom:7.1f} MB   -> {verdict}")

    if headroom <= 0:
        print()
        print("Fixes, in order of preference:")
        print("  1. Reduce the catalogue. Embeddings are the cheap part of memory;")
        print("     MiniLM is not, so this buys little.")
        print("  2. Use ONNX Runtime instead of torch (~50MB vs ~250MB+).")
        print("  3. Move to a paid tier. A $7/mo instance removes the question.")
        sys.exit(1)


if __name__ == "__main__":
    main()