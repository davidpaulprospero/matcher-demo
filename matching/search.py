"""Embedding search over the clip catalogue.

Two decisions live here, and both came from watching the plain version fail.

1. ADAPTIVE POOL SIZE (`_complexity_score`)
   A fixed k is wrong in both directions. "It works." is short and generic —
   almost any clip matches, so a narrow pool throws away the best available
   option. "In late March of nineteen ninety-four the committee published its
   findings." is long and specific — a narrow pool is fine and a wide one
   mostly returns noise. So k scales *inversely* with complexity: generic lines
   get a wide pool to choose from, specific lines get a tight one.

2. PRE-FETCH THEN DIVERSIFY (`max_per_source`)
   Plain top-k is dominated by whichever source has the most clips. Ten
   near-identical beach shots will occupy all ten slots, and a 20-clip shoot
   outnumbers a 24-clip one purely by luck of the draw. The fix is to over-fetch
   (k x pre_fetch_multiplier) and then take at most
   `max_candidates_per_source` per source while preserving score order.

   Under-fetching instead — asking FAISS for exactly k and capping after — is the
   obvious version and it does not work: once the top k are all one source there
   is nothing left to diversify from, and the only fix is to grow k again. This
   is why over-fetch is the default and `pre_fetch_multiplier` exists at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class SearchConfig:
    embedding_candidates: int = 20
    max_candidates_per_source: int = 3
    min_candidates: int = 15
    max_candidates: int = 100
    complexity_scaling_factor: float = 0.5
    pre_fetch_multiplier: int = 2

    def from_dict(cls_kwargs: Dict) -> "SearchConfig":
        """Build from a dict, ignoring unknown keys."""
        known = {f for f in cls_kwargs.__dataclass_fields__}
        return cls(**{k: v for k, v in cls_kwargs.items() if k in known})


@dataclass
class Candidate:
    """A clip and how well it matched."""

    clip: Dict
    similarity: float
    # Recorded for the UI so a bad match can be explained rather than just
    # shown: how much of the score came from reuse penalties.
    penalty_applied: float = 0.0
    pool_size: int = 0

    @property
    def base_similarity(self) -> float:
        return min(1.0, self.similarity + self.penalty_applied)


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of one vector against a matrix of vectors.

    Matrices are L2-normalised once by the caller when there is more than one
    query, but normalising here keeps the single-query path honest.
    """
    if matrix.size == 0:
        return np.zeros(0, dtype=np.float32)

    q = query / (np.linalg.norm(query) or 1.0)
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1.0
    return (matrix @ q) / norms


# Words that signal a place. A line naming a specific place is hard to satisfy,
# so it should not get a wide pool full of generic options.
_LOCATION_WORDS = frozenset({
    "city", "town", "village", "country", "state", "province", "region",
    "mountain", "river", "lake", "ocean", "sea", "island", "beach",
    "street", "road", "bridge", "park", "building", "tower", "castle",
    "cathedral", "temple", "mosque", "church", "monument", "palace",
    "museum", "harbor", "harbour", "port", "airport", "station", "square", "plaza",
    "valley", "ridge", "field", "path", "coast", "shore", "shoreline", "cliff",
})


def complexity_score(text: str) -> float:
    """How specific a voiceover line is, from 0.0 (generic) to 1.0 (specific).

    Three signals, weighted by how reliably each predicts "this line will be
    hard to match":

    - length (0.40) — the strongest signal, and the cheapest.
    - capitalised non-initial words (0.35) — a proxy for named entities.
    - location words (0.25) — geographic specificity.

    Caps are the point. Everything saturates: a 40-word line is not four times
    harder to match than a 10-word one, so the word factor stops at 15 words
    rather than scaling without limit.
    """
    if not text or not text.strip():
        return 0.0

    words = text.split()
    count = len(words)

    if count <= 3:
        word_factor = 0.0
    elif count >= 15:
        word_factor = 1.0
    else:
        word_factor = (count - 3) / 12.0

    # Capitalised words that are not sentence-initial.
    entity_count = sum(
        1 for i, w in enumerate(words) if i > 0 and w[:1].isupper() and len(w) > 1
    )
    entity_factor = min(entity_count / 3.0, 1.0)

    location_count = len({w.lower().strip(".,!?") for w in words} & _LOCATION_WORDS)
    location_factor = min(location_count / 2.0, 1.0)

    score = 0.4 * word_factor + 0.35 * entity_factor + 0.25 * location_factor
    return float(min(max(score, 0.0), 1.0))


def pool_size_for(text: str, config: SearchConfig) -> int:
    """Adaptive candidate pool size for one line.

    Inverse scaling: complexity 0.0 gives the full `1 + factor` boost,
    complexity 1.0 gives none.
    """
    base = config.embedding_candidates
    scaled = base * (1.0 + config.complexity_scaling_factor * (1.0 - complexity_score(text)))
    return int(max(config.min_candidates, min(config.max_candidates, scaled)))


def diversify(
    scored: Sequence[Tuple[int, float]],
    source_of: Callable[[int], str],
    k: int,
    max_per_source: int,
) -> List[Tuple[int, float]]:
    """Take the best `k` while capping how many come from any one source.

    `scored` must be sorted by score descending. Preserves that order in the
    output, so the result is still ranked.

    Note the failure mode: if `max_per_source` is so tight that every source
    hits its cap before `k` is reached, the result is *shorter* than k. That is
    correct — padding it back with duplicates of the top source is exactly the
    concentration this is preventing. The caller is told the real pool size so
    the UI can say so.
    """
    if max_per_source <= 0:
        return list(scored[:k])

    taken: Dict[str, int] = {}
    out: List[Tuple[int, float]] = []

    for idx, score in scored:
        source = source_of(idx)
        if taken.get(source, 0) >= max_per_source:
            continue
        taken[source] = taken.get(source, 0) + 1
        out.append((idx, score))
        if len(out) >= k:
            break

    return out


class EmbeddingSearch:
    """Ranks catalogue clips against one voiceover line."""

    def __init__(
        self,
        clips: List[Dict],
        embeddings: np.ndarray,
        config: Optional[SearchConfig] = None,
    ):
        self.clips = clips
        self.config = config or SearchConfig()
        self.embeddings = embeddings
        self._by_id = {c["id"]: i for i, c in enumerate(clips)}

    def source_of(self, index: int) -> str:
        return self.clips[index]["source"]

    def search(self, query_embedding: np.ndarray, voiceover_text: str) -> List[Candidate]:
        """Ranked candidates for one line.

        Applies the over-fetch and diversity cap; reuse penalties are applied
        later by the pipeline, because they depend on what has already been
        placed and the search itself is stateless.
        """
        if len(self.clips) == 0:
            return []

        k = pool_size_for(voiceover_text, self.config)
        fetch_k = k * self.config.pre_fetch_multiplier

        # Catalogue may be shorter than the fetch size; that's fine.
        fetch_k = min(fetch_k, len(self.clips))

        scores = cosine_similarity(query_embedding, self.embeddings)
        # argpartition avoids sorting the whole corpus; only the top slice
        # needs to be ordered.
        top_idx = np.argpartition(-scores, fetch_k - 1)[:fetch_k]
        ranked = sorted(
            ((int(i), float(scores[i])) for i in top_idx),
            key=lambda x: x[1],
            reverse=True,
        )

        picked = diversify(
            ranked, self.source_of, k, self.config.max_candidates_per_source
        )
        return [
            Candidate(clip=self.clips[i], similarity=score, pool_size=len(picked))
            for i, score in picked
        ]