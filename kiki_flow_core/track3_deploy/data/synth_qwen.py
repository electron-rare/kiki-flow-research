"""Generate species-aware French queries via Qwen3.5-35B over an OpenAI-compatible HTTP tunnel.

Species keys are SHORT names ("phono", "sem", "lex", "syntax") used as corpus tags
and CorpusEntry.species field. The mapping to canonical JKO keys ("<short>:code")
happens at the JKO-oracle boundary, not here.
"""

from __future__ import annotations

import re
from typing import Any

import httpx

from kiki_flow_core.track3_deploy.data.corpus_builder import CorpusEntry

SPECIES_PROMPTS: dict[str, str] = {
    "phono": (
        "Génère une query courte en français qui sollicite fortement le traitement "
        "phonologique (mots rares, homophones, assonances, phonèmes difficiles). "
        "Longueur 8-24 mots. Une query par ligne, sans numérotation."
    ),
    "sem": (
        "Génère une query en français impliquant désambiguïsation sémantique, "
        "polysémie, relations lexicales (synonymes, hyperonymes, antonymes). "
        "Longueur 8-24 mots. Une query par ligne, sans numérotation."
    ),
    "lex": (
        "Génère une query en français avec mots de basse fréquence, néologismes, "
        "registre spécialisé (technique, littéraire, scientifique). "
        "Longueur 8-24 mots. Une query par ligne, sans numérotation."
    ),
    "syntax": (
        "Génère une query en français avec structure syntaxique complexe : "
        "dépendances longues, subordonnées imbriquées, ambiguïtés d'attachement. "
        "Longueur 12-32 mots. Une query par ligne, sans numérotation."
    ),
}

_NUMBERED_PREFIX = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+")

_DEFAULT_BATCH_SIZE = 50
_DEFAULT_TEMPERATURE = 0.8
_DEFAULT_TOP_P = 0.9
_DEFAULT_TIMEOUT_SEC = 120.0
_TOKENS_PER_QUERY = 50  # heuristic for max_tokens budget


def _parse_lines(content: str) -> list[str]:
    out: list[str] = []
    for raw in content.splitlines():
        line = _NUMBERED_PREFIX.sub("", raw).strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


class SyntheticGenerator:
    """Wrapper around Qwen3.5-35B OpenAI-compatible endpoint for synthetic corpus D."""

    def __init__(
        self,
        base_url: str = "http://localhost:18000",
        model: str = "Qwen3.5-35B-A3B-UD-Q3_K_XL",
        batch_size: int = _DEFAULT_BATCH_SIZE,
        temperature: float = _DEFAULT_TEMPERATURE,
        top_p: float = _DEFAULT_TOP_P,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.batch_size = batch_size
        self.temperature = temperature
        self.top_p = top_p
        self.client = client or httpx.Client(timeout=_DEFAULT_TIMEOUT_SEC)

    def _call(self, prompt: str, n: int) -> list[str]:
        """One HTTP call; returns parsed queries."""
        url = f"{self.base_url}/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": f"{prompt}\n\nGénère {n} queries."}],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": _TOKENS_PER_QUERY * n,
        }
        resp = self.client.post(url, json=payload)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return _parse_lines(content)

    def generate_batch(self, species: str, n: int) -> list[str]:
        """Accumulate queries across multiple calls until we have n unique ones."""
        if species not in SPECIES_PROMPTS:
            raise ValueError(
                f"Unknown species: {species!r}; expected one of {sorted(SPECIES_PROMPTS.keys())}"
            )
        prompt = SPECIES_PROMPTS[species]
        seen: set[str] = set()
        out: list[str] = []
        while len(out) < n:
            batch = self._call(prompt, min(self.batch_size, n - len(out) + 5))
            for q in batch:
                if q in seen:
                    continue
                seen.add(q)
                out.append(q)
                if len(out) >= n:
                    break
        return out[:n]

    def generate_tagged(self, species: str, n: int) -> list[CorpusEntry]:
        """Return queries wrapped as CorpusEntry(source='D', species=...)."""
        return [
            CorpusEntry(text=q, source="D", species=species)
            for q in self.generate_batch(species, n)
        ]
