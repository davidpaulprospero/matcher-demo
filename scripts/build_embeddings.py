"""Precompute catalogue embeddings.

Run once, offline, and commit the result:

    python scripts/build_embeddings.py

This takes a couple of minutes on CPU for ~240 clips. Doing it here rather
than at app start is the point: a free host that sleeps after 15 idle minutes
would otherwise pay model load + encode on every cold start, and the first
visitor would wait a minute for results.

Re-run this whenever scripts/generate_corpus.py changes. The file records the
model name, and the app refuses to load embeddings whose model does not match.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from matching.embeddings import MODEL_NAME, embed, load_catalog, save_embeddings  # noqa: E402

CATALOG = ROOT / "data" / "catalog.json"
OUTPUT = ROOT / "data" / "embeddings.json"


def main() -> None:
    catalog = load_catalog(CATALOG)
    clips = catalog["clips"]

    print(f"{len(clips)} clips from {CATALOG.relative_to(ROOT)}")
    print(f"model: {MODEL_NAME}")

    vectors = embed([c["caption"] for c in clips])
    save_embeddings(OUTPUT, vectors, [c["id"] for c in clips])

    size_mb = OUTPUT.stat().st_size / 1e6
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({size_mb:.2f} MB)")
    print(f"shape: {vectors.shape}")


if __name__ == "__main__":
    main()