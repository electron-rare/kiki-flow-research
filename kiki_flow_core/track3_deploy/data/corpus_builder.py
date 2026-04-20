"""Assemble, dedup, and stratify-split the hybrid corpus for text-bridge training."""

from __future__ import annotations

import hashlib
import logging
import random
import re
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

Embedder = Callable[[list[str]], np.ndarray]  # (n, D) float32, normalized


@dataclass(frozen=True)
class CorpusEntry:
    text: str
    source: str  # "B", "C", or "D"
    species: str  # "phono", "sem", "lex", "syntax"  — short names; map to canonical at JKO boundary


# Lower priority number = kept on cross-source dup (B > C > D).
_SOURCE_PRIORITY: dict[str, int] = {"B": 0, "C": 1, "D": 2}

# Minimum stage-1 survivors to bother running embedding dedup.
_MIN_FOR_EMBED_DEDUP = 2

# Tolerance for ratios-sum check.
_RATIO_SUM_TOLERANCE = 1e-6


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace, NFKD."""
    normalized = unicodedata.normalize("NFKD", text).lower()
    normalized = re.sub(r"[^\w\s]", "", normalized, flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


class CorpusBuilder:
    """Build train/val/test splits from a hybrid corpus.

    Dedup policy:
      1. Exact match on normalized text → drop duplicates, keep earliest.
      2. Embedding dedup (cosine > threshold) → drop the lower-priority
         source on cross-source dup; otherwise drop the shorter text.
         Embeddings are provided externally via the ``embedder`` callable.

    Args:
        dedup_threshold: Cosine similarity above which two entries are
            considered near-duplicates. Default 0.92.
        embedder: Optional callable ``(list[str]) -> np.ndarray`` of shape
            ``(n, D)`` float32, normalized. If ``None``, only exact-match
            dedup runs (a warning is logged). Typical providers: MLX port
            of MiniLM, pre-computed .npz cache, or a closure over an
            external model that is free of torch.
    """

    def __init__(
        self,
        dedup_threshold: float = 0.92,
        embedder: Embedder | None = None,
    ) -> None:
        self.dedup_threshold = dedup_threshold
        self.embedder = embedder

    @staticmethod
    def _resolve_dup(
        stage1: list[CorpusEntry],
        keep: list[bool],
        i: int,
        j: int,
    ) -> bool:
        """Mark one of i/j for removal. Returns True if i was dropped (break outer)."""
        pi = _SOURCE_PRIORITY[stage1[i].source]
        pj = _SOURCE_PRIORITY[stage1[j].source]
        if pi < pj:
            keep[j] = False
            return False
        if pj < pi:
            keep[i] = False
            return True
        # same source — drop shorter
        if len(stage1[i].text) >= len(stage1[j].text):
            keep[j] = False
            return False
        keep[i] = False
        return True

    def dedup_exact(self, entries: Iterable[CorpusEntry]) -> list[CorpusEntry]:
        """Stage 1: exact match on normalized text. Always runs."""
        seen_norm: set[str] = set()
        result: list[CorpusEntry] = []
        for e in entries:
            key = _normalize(e.text)
            if key in seen_norm:
                continue
            seen_norm.add(key)
            result.append(e)
        return result

    def dedup_by_embeddings(
        self,
        entries: list[CorpusEntry],
        embeddings: np.ndarray,
    ) -> list[CorpusEntry]:
        """Stage 2: cosine-similarity dedup using PROVIDED embeddings.

        ``entries[i]`` corresponds to ``embeddings[i]``. The caller computes
        embeddings separately (e.g. MLX port of MiniLM, pre-computed .npz
        cache). No ML dependency is introduced here.
        """
        if len(entries) < _MIN_FOR_EMBED_DEDUP:
            return entries
        keep = [True] * len(entries)
        for i in range(len(entries)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(entries)):
                if not keep[j]:
                    continue
                if _cosine(
                    embeddings[i], embeddings[j]
                ) > self.dedup_threshold and self._resolve_dup(entries, keep, i, j):
                    break
        return [e for e, k in zip(entries, keep, strict=True) if k]

    def dedup(self, entries: Iterable[CorpusEntry]) -> list[CorpusEntry]:
        """Full dedup: exact stage + optional embedding stage.

        If ``self.embedder`` is ``None``, only exact dedup runs (with a
        warning logged).
        """
        stage1 = self.dedup_exact(entries)
        if self.embedder is None:
            logging.getLogger(__name__).warning(
                "CorpusBuilder.dedup: no embedder provided, skipping embedding stage. "
                "Only exact-match dedup performed."
            )
            return stage1
        embs = self.embedder([e.text for e in stage1])
        return self.dedup_by_embeddings(stage1, embs)

    def split(
        self,
        entries: list[CorpusEntry],
        ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 0,
    ) -> dict[str, list[CorpusEntry]]:
        """Stratified split by (source, species), deterministic on seed."""
        if abs(sum(ratios) - 1.0) > _RATIO_SUM_TOLERANCE:
            raise ValueError(f"ratios must sum to 1.0, got {sum(ratios)}")
        strata: dict[tuple[str, str], list[CorpusEntry]] = {}
        for e in entries:
            strata.setdefault((e.source, e.species), []).append(e)
        out: dict[str, list[CorpusEntry]] = {"train": [], "val": [], "test": []}
        rng = random.Random(seed)
        for _key, bucket in strata.items():
            items = list(bucket)
            rng.shuffle(items)
            n = len(items)
            n_train = int(n * ratios[0])
            n_val = int(n * ratios[1])
            out["train"].extend(items[:n_train])
            out["val"].extend(items[n_train : n_train + n_val])
            out["test"].extend(items[n_train + n_val :])
        return out

    @staticmethod
    def freeze_hash(entries: list[CorpusEntry]) -> str:
        """Deterministic hash of a corpus split for auditability."""
        joined = "\n".join(f"{e.source}|{e.species}|{e.text}" for e in entries)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()
