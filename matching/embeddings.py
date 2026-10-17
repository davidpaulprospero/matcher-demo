"""Embedding provider.

Why ONNX and not PyTorch
------------------------
The private pipeline this is drawn from uses `sentence-transformers`, which
needs torch. That does not fit the deployment target, and the failure would be
bad: Render's free instance has 512MB, and torch plus the model measures 555MB
resident — over the limit before the first request.

Measured on this machine via `scripts/check_memory.py`:

    all-MiniLM-L6-v2 via sentence-transformers (torch)   555.4 MB   DOES NOT FIT
    BAAI/bge-small-en-v1.5 via ONNX Runtime (fastembed)   245.9 MB   fits, 266MB spare

So this uses ONNX Runtime. Both models are 384-dimensional sentence encoders of
similar size (22M and 33M parameters), so the matching behaviour is comparable;
the ~310MB saving is what makes the free tier possible.

Neither model needs an API key, a daemon, or a network call at inference time.
That is the other half of the decision: an unpaid key behind a public URL is an
open tab for anyone, and a metered one sends the bill regardless of traffic.

Catalogue embeddings are computed once by `scripts/build_embeddings.py` and
committed, so a cold start is "load 1MB of JSON + load a 130MB model" rather
than "load a model and encode 269 clips".
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Sequence

import numpy as np

# bge-small-en-v1.5: 384-dim, 33M params, ~130MB in ONNX int8.
MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

# Where the ONNX weights live. Inside the project so it is visible and can be
# pointed elsewhere by env var, and so a wiped container reuses nothing stale.
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / ".model_cache"

_model = None


def get_model():
    """Load the model on first use, then keep it for the process lifetime.

    A Streamlit server reuses one process across sessions, so this is a dict
    lookup after the first request rather than a 130MB load every time.

    The cache directory is explicit. Left to itself fastembed picks a per-user
    temp path, which is fine locally and unpredictable on a host — and the
    cache location is what decides whether a woken container reuses a warm
    model or downloads ~65MB again before it can answer anything.
    """
    global _model
    if _model is None:
        from fastembed import TextEmbedding

        cache = Path(os.environ.get("FASTEMBED_CACHE_DIR", DEFAULT_CACHE_DIR))
        cache.mkdir(parents=True, exist_ok=True)

        _model = TextEmbedding(MODEL_NAME, cache_dir=str(cache))
    return _model


def embed(texts: Sequence[str]) -> np.ndarray:
    """Embed strings into an (n, 384) float32 matrix.

    L2-normalised, so a dot product is the cosine similarity — which is the
    assumption `search.cosine_similarity` makes on its fast path.
    """
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    vectors = np.array(list(get_model().embed(list(texts))), dtype=np.float32)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # A zero vector would otherwise produce NaN and silently poison the search.
    norms[norms == 0] = 1.0
    return vectors / norms


def load_catalog(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not data.get("synthetic"):
        # Loud, not silent. If this is ever pointed at real licensed metadata,
        # the UI must not be able to go on claiming the data is synthetic.
        raise ValueError(
            "Expected a synthetic catalogue; this one is not marked synthetic."
        )
    return data


def load_embeddings(path: Path) -> np.ndarray:
    """Load the committed catalogue embeddings, refusing a model mismatch."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("model") != MODEL_NAME:
        raise ValueError(
            f"Embeddings were built with {raw.get('model')!r} but this app "
            f"uses {MODEL_NAME!r}. Re-run scripts/build_embeddings.py."
        )
    return np.asarray(raw["vectors"], dtype=np.float32)


def save_embeddings(
    path: Path, vectors: np.ndarray, clip_ids: List[str], model: str = MODEL_NAME
) -> None:
    """Write embeddings to JSON.

    JSON rather than .npy so the file is reviewable in a diff and needs no extra
    dependency to inspect. At 384 floats per clip for 269 clips that is ~1MB,
    which is a fair price for being able to actually read what is in the repo.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "dim": int(vectors.shape[1]),
        "clip_ids": clip_ids,
        "vectors": [[round(float(v), 6) for v in row] for row in vectors],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")