"""CLI: read JSONL corpus, run HeuristicLabeler, save per-query labels to NPZ.

Checkpointed variant of label_corpus.py — conforms to scripts/CLAUDE.md
anti-pattern rule: "Long-running sweeps with no checkpointing. If the
script takes more than ~10 min, write incremental partial results so a
crash doesn't destroy N hours of compute."

Output format is identical to label_corpus.py: a single .npz keyed by
sha256(query). Intermediate .part{N}.npz files are merged and removed
at the end. On SIGINT/SIGTERM, the latest part stays on disk so a
restart can read through it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np

SPECIES_CANONICAL = ("phono:code", "sem:code", "lex:code", "syntax:code")
LOG_EVERY = 500
BATCH = 500
logger = logging.getLogger(__name__)


def _flush(batch: dict[str, np.ndarray], output: Path, counter: int) -> Path:
    part = output.with_suffix(f".part{counter}.npz")
    np.savez_compressed(part, **batch)
    logger.info("flushed part %s (%d entries)", part.name, len(batch))
    return part


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lexique", type=Path, default=None)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    from kiki_flow_core.track3_deploy.data.heuristic_labeler import (  # noqa: PLC0415
        HeuristicLabeler,
    )

    labeler = HeuristicLabeler(lexique_csv=args.lexique)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    batch: dict[str, np.ndarray] = {}
    parts: list[Path] = []
    n = 0
    with args.corpus.open() as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            entry = json.loads(stripped)
            q = entry["text"]
            h = hashlib.sha256(q.encode("utf-8")).hexdigest()
            labels = labeler.label(q)
            stacked = np.stack([labels[sp] for sp in SPECIES_CANONICAL]).astype(np.float32)
            batch[h] = stacked
            n += 1
            if n % LOG_EVERY == 0:
                logger.info("labeled %d queries", n)
            if n % args.batch == 0:
                parts.append(_flush(batch, args.output, n))
                batch = {}
    if batch:
        parts.append(_flush(batch, args.output, n))

    logger.info("merging %d parts into %s", len(parts), args.output)
    merged: dict[str, np.ndarray] = {}
    for p in parts:
        with np.load(p) as f:
            for k in f.files:
                merged[k] = f[k]
    np.savez_compressed(args.output, **merged)
    for p in parts:
        p.unlink()
    logger.info("wrote %d labels to %s", n, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
