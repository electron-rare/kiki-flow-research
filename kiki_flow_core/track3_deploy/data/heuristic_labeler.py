"""HeuristicLabeler — produces per-species simplex labels from FR query text.

Phase-A pre-training target generator. Uses SpaCy-fr for tokenization/parse,
phonemizer for phoneme extraction, optional Lexique.org for frequency bins.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from kiki_flow_core.track3_deploy.data.phono_classes import (
    DEFAULT_CLASS,
    PHONO_CLASSES,
)
from kiki_flow_core.track3_deploy.data.sem_categories import N_SEM
from kiki_flow_core.track3_deploy.data.syntax_patterns import (
    N_SYNTAX,
    SYNTAX_PATTERNS,
)

N_STACKS = 32
_EPS_SMOOTH = 1e-6
_DEFAULT_SPACY_MODEL = "fr_core_news_lg"
_DEFAULT_LEX_BIN = N_STACKS // 2

# macOS Homebrew espeak-ng dylib paths (tried in order)
_MACOS_ESPEAK_LIBS = [
    "/opt/homebrew/lib/libespeak-ng.dylib",
    "/usr/local/lib/libespeak-ng.dylib",
    "/opt/homebrew/lib/libespeak-ng.1.dylib",
]


def _configure_espeak_library() -> None:
    """On macOS, phonemizer cannot auto-detect the Homebrew espeak-ng dylib.

    This function sets the library path explicitly if running on macOS and
    the default detection fails.
    """
    if sys.platform != "darwin":
        return
    try:
        from phonemizer.backend.espeak.wrapper import EspeakWrapper  # noqa: PLC0415

        # Try a quick instantiation; if it raises, set the library manually
        EspeakWrapper()
    except RuntimeError:
        from phonemizer.backend.espeak.wrapper import EspeakWrapper  # noqa: PLC0415

        for lib_path in _MACOS_ESPEAK_LIBS:
            if Path(lib_path).exists():
                EspeakWrapper.set_library(lib_path)
                break


def _uniform_simplex() -> np.ndarray:
    return np.ones(N_STACKS, dtype=np.float32) / N_STACKS


def _smooth_normalize(counts: np.ndarray) -> np.ndarray:
    """Add-epsilon smoothing then L1 normalize."""
    smoothed = counts + _EPS_SMOOTH
    out: np.ndarray = (smoothed / smoothed.sum()).astype(np.float32)
    return out


class HeuristicLabeler:
    """Produce per-species 32-simplex labels for FR query text."""

    def __init__(
        self,
        spacy_model: str = _DEFAULT_SPACY_MODEL,
        lexique_csv: Path | str | None = None,
        phoneme_lang: str = "fr-fr",
    ) -> None:
        import spacy  # noqa: PLC0415

        self._nlp = spacy.load(spacy_model, disable=["ner"])
        self._phoneme_lang = phoneme_lang
        self._lexique_bins: dict[str, int] | None = None
        _configure_espeak_library()
        # Preload the espeak backend once. The top-level `phonemize()` helper
        # builds a fresh EspeakBackend on every call, which copies the
        # espeak-ng shared library into /tmp each time — on a long sweep
        # (e.g. 8k corpus labelling) this saturates mmap and crashes with
        # "échec d'adressage du segment de l'objet partagé".
        from phonemizer.backend import EspeakBackend  # noqa: PLC0415

        self._phoneme_backend = EspeakBackend(language=phoneme_lang, preserve_punctuation=False)
        if lexique_csv is not None:
            self._load_lexique(Path(lexique_csv))

    def _load_lexique(self, path: Path) -> None:
        """Load Lexique.org CSV, precompute log-freq bin (0-31) per lemma."""
        import pandas as pd  # noqa: PLC0415

        df = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
        freq_col = "freqlivres" if "freqlivres" in df.columns else df.columns[1]
        lemma_col = "lemme" if "lemme" in df.columns else df.columns[0]
        freqs = df[freq_col].fillna(0.0).astype(float).clip(lower=_EPS_SMOOTH)
        log_freqs = np.log1p(freqs)
        q = np.linspace(0, 1, N_STACKS + 1)[1:-1]
        edges = np.quantile(log_freqs, q)
        bins = np.digitize(log_freqs, edges)
        self._lexique_bins = dict(
            zip(df[lemma_col].astype(str).str.lower(), bins.tolist(), strict=False)
        )

    def label(self, query: str) -> dict[str, np.ndarray]:
        if not query.strip():
            return {
                "phono:code": _uniform_simplex(),
                "sem:code": _uniform_simplex(),
                "lex:code": _uniform_simplex(),
                "syntax:code": _uniform_simplex(),
            }
        doc = self._nlp(query)
        return {
            "phono:code": self._phono_distribution(query),
            "sem:code": self._sem_distribution(doc),
            "lex:code": self._lex_distribution(doc),
            "syntax:code": self._syntax_distribution(doc),
        }

    def _phono_distribution(self, query: str) -> np.ndarray:
        ipa_list = self._phoneme_backend.phonemize([query], strip=True)
        ipa = ipa_list[0] if ipa_list else ""
        counts = np.zeros(N_STACKS, dtype=np.float32)
        for ch in ipa:
            cls = PHONO_CLASSES.get(ch, DEFAULT_CLASS)
            counts[cls] += 1.0
        return _smooth_normalize(counts)

    def _sem_distribution(self, doc: Any) -> np.ndarray:
        """Heuristic: map content lemmas to category bins via stable hash.

        Placeholder for WordNet-fr; sem clustering refinement deferred to v0.4.
        """
        counts = np.zeros(N_STACKS, dtype=np.float32)
        for tok in doc:
            if tok.pos_ in ("NOUN", "VERB", "ADJ", "ADV") and not tok.is_stop:
                idx = abs(hash(tok.lemma_.lower())) % N_SEM
                counts[idx] += 1.0
        return _smooth_normalize(counts)

    def _lex_distribution(self, doc: Any) -> np.ndarray:
        counts = np.zeros(N_STACKS, dtype=np.float32)
        for tok in doc:
            if tok.is_alpha:
                if self._lexique_bins is not None:
                    bin_idx = self._lexique_bins.get(tok.lemma_.lower(), _DEFAULT_LEX_BIN)
                else:
                    bin_idx = abs(hash(tok.text.lower())) % N_STACKS
                counts[int(bin_idx)] += 1.0
        return _smooth_normalize(counts)

    def _syntax_distribution(self, doc: Any) -> np.ndarray:
        dep_to_idx = {dep: i for i, dep in enumerate(SYNTAX_PATTERNS)}
        counts = np.zeros(N_STACKS, dtype=np.float32)
        fallback_idx = N_SYNTAX - 1
        for tok in doc:
            idx = dep_to_idx.get(tok.dep_, fallback_idx)
            counts[idx] += 1.0
        return _smooth_normalize(counts)
