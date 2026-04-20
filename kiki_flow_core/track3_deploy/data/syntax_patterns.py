"""32 FR syntactic patterns based on Universal Dependency labels (used by SpaCy-fr)."""

from __future__ import annotations

SYNTAX_PATTERNS: tuple[str, ...] = (
    "nsubj",
    "obj",
    "iobj",
    "obl",
    "csubj",
    "ccomp",
    "xcomp",
    "nmod",
    "amod",
    "appos",
    "advmod",
    "det",
    "case",
    "mark",
    "cc",
    "conj",
    "aux",
    "cop",
    "acl",
    "advcl",
    "parataxis",
    "list",
    "root",
    "punct",
    "compound",
    "fixed",
    "flat",
    "nummod",
    "expl",
    "discourse",
    "vocative",
    "dep",
)

N_SYNTAX = 32

assert len(SYNTAX_PATTERNS) == N_SYNTAX
