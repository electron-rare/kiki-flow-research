# Cross-paper review notes — 2026-04-22 night

Late-night critic pass after committing the workshop v2 promotion and the
asym-drift systematic finding. Compares the main paper (`paper/main.tex`)
against the workshop preprint (`paper/workshop/main.tex` +
`section_[1-6]_*.tex`) and flags inconsistencies that a reviewer reading
both would catch. Each finding has a severity rating (CRITICAL / HIGH /
MEDIUM / LOW) and a concrete suggested fix.

To attend with a fresh mind. Not blocking for the current repo state.

---

## CRITICAL

### C1. Species ordering mismatch between workshop prose and code

- **Code ground truth**: `kiki_flow_core/species/data/dell_baddeley_coupling.yaml:19`
  declares `species_order: [phono, lex, syntax, sem]`, and
  `kiki_flow_core/species/canonical_species.py` loads that order
  verbatim. The same order appears in the main paper (§3, §5 table).
- **Workshop claim**: `paper/workshop/section_3_method_overview.tex:5-7`
  writes the state tuple as
  $\rho = (\rho_{\mathrm{phono}}, \rho_{\mathrm{sem}}, \rho_{\mathrm{lex}}, \rho_{\mathrm{syntax}})$.
  The abstract (`paper/workshop/main.tex`) similarly lists
  "phonological, semantic, lexical, syntactic".
- **Impact**: a reviewer who maps the workshop prose indices onto the
  code's `J_{ij}` matrix would pick up the wrong coupling coefficients
  (`J_{lex \leftarrow sem}` in code is not `J_{sem \leftarrow lex}` in
  prose). Reproduction from the workshop alone fails silently.
- **Fix**: change workshop prose to
  `(ρ_phono, ρ_lex, ρ_syntax, ρ_sem)` to match code. Simplest
  (sed-safe) path since the code + main paper are the canonical source.

---

## HIGH

### H1. Workshop J-coupling term lags behind main paper §3 decomposition

- **Workshop state**: `section_3_method_overview.tex:46` and
  `section_4_method_v2.tex` write the coupling term as
  $\lambda_J \sum_{s,t} J_{st} \langle \rho_s, \rho_t \rangle$ with no
  mention of the $J = J^{\mathrm{sym}} + J^{\mathrm{asym}}$ split or
  the fact that only $J^{\mathrm{sym}}$ contributes to the scalar
  energy. The main paper §3 (`paper/main.tex:133-163`) is now explicit
  on this.
- **Impact**: a reader of both papers sees two different formalisms for
  the same coupling. Workshop looks stale.
- **Fix**: add one sentence to workshop §3 or §4 citing the main-paper
  decomposition, e.g. *"we follow the $J = J^{\mathrm{sym}} +
  J^{\mathrm{asym}}$ decomposition of \S3 of~\cite{saillant2026kikiflow}
  and route the scalar energy through $J^{\mathrm{sym}}$ only."*

### H2. Grid dimensionality mismatch (32 bins vs 16 bins)

- **Workshop**: 32 bins per species (so product space is
  $(\Delta^{32})^4$, i.e. 128-dim flat); see
  `section_3_method_overview.tex:6-8` and 58-60.
- **Main paper**: 16 bins per species
  (`paper/main.tex:334` — "on a reduced problem (grid 16, 30 slow
  steps)"; confirmed in `paper/hyperparam_sweep_coupling_modes.json`
  meta: `grid: 16`).
- **Impact**: same species formalism but different resolution is
  legitimate (workshop wants more resolution for the 32-stack semantic
  mapping); the confusion is that both papers say "four-species
  Levelt-Baddeley simplex" with no disambiguation of the bin count.
- **Fix**: add a one-line disambiguation sentence in each paper. Main:
  *"at grid 16 (`kiki_flow_core` default for the T2 track)"*. Workshop:
  *"at grid 32 for the QueryConditionedF oracle — see `method-overview`
  for rationale"*. No equation changes, just labels.

### H3. Citation divergence on Levelt / Baddeley

- **Main paper** (`paper/references.bib`): `levelt1999` (BBS 22),
  `baddeley1992working` (Science 255).
- **Workshop** (`paper/workshop/references.bib`, inferred from
  `section_3_method_overview.tex:65` which cites
  `levelt1989speaking, baddeley1974`): Levelt 1989 (Speaking book),
  Baddeley 1974 (working-memory original).
- **Impact**: same authors, different works, different years. A careful
  reviewer spots the inconsistency and asks which is canonical for
  the Levelt-Baddeley architecture claim. Both are legitimate, but
  mixing them without explanation looks sloppy.
- **Fix**: pick one policy --- either cite both works side-by-side in
  both papers, or use the 1999 / 1992 pair in both (the 1999 BBS
  article is the more operational formulation for production).

---

## MEDIUM

### M1. Narrative tension on "what coupling does"

- **Main paper** (new §5 + §6): specialization is
  *potential-driven*, not coupling-driven; the ablation shows
  `separable` slightly beats `dell-full` at the canonical operating
  point; asymmetric drift is uniformly anti-specializing.
- **Workshop** (§1 intro): *"T1 and T2 are query-agnostic ... defeating
  the purpose of training a text-aware downstream model.
  QueryConditionedF is the minimal modification that makes the oracle
  react to the query."*
- **Potential reading**: if coupling doesn't matter for specialization
  (main), why does the workshop invest in a more elaborate coupling?
- **Resolution**: the two papers are experimentally in different
  regimes. Main tests the stationary entropy under query-agnostic
  setup; workshop tests trajectory divergence under query conditioning.
  The workshop's query term is not another `J` coupling — it's a
  reconstruction-loss-based tether that couples `ρ` to an external
  embedding.
- **Fix**: add one paragraph to workshop §6 (or §1 "Positioning") that
  explicitly resolves this: *"The companion query-agnostic paper
  \cite{saillant2026kikiflow} shows that the J coupling alone does
  not drive stationary specialization ... The query-conditioning term
  of QueryConditionedF is mechanistically distinct ..."*.
  Closes the reviewer risk cleanly.

### M2. No cross-citation between the two papers

- Neither paper currently cites the other. If submitted together or
  close in time, they should. The main paper in particular could cite
  the workshop as the query-conditioned extension, and vice versa.
- **Fix**: add bib entry `saillant2026querycondf` (unpublished,
  workshop) to main paper's `references.bib` and cite in main §6 under
  "future work extends this to query-conditioned oracles". Symmetrically
  add `saillant2026kikiflow` to workshop and cite in §1 Positioning.

### M3. Pipeline figure labels stale terminology

- `paper/workshop/section_3_method_overview.tex:43-46` (pipeline
  figure) writes `∇λ_J Σ J_st ⟨ρ_s, ρ_t⟩`. This is legal but now
  has the same issue as H1: no split sym/asym.
- **Fix**: update the figure caption or equation to reflect the split,
  or add a small footnote.

### M4. Active-inference framing absent from main paper

- Workshop (§1) grounds the work explicitly in Friston 2010 / Parr
  2017 active-inference, with accuracy + complexity decomposition.
- Main paper uses Wasserstein gradient flows + Conger-Hoffmann without
  this framing.
- **Impact**: the two papers read as entirely different framings of
  overlapping work. A reviewer used to active-inference (neuroscience
  audience) would expect main paper to reference Friston too; a
  reviewer from the Wasserstein community would ask why workshop uses
  active inference at all.
- **Fix**: main paper §6 could add one sentence mentioning that the
  QueryConditionedF extension casts the machinery as active
  inference; workshop could add one sentence acknowledging that the
  query-agnostic core is a standard multi-species Wasserstein flow.

---

## LOW

### L1. "Stacks" terminology divergence

- Workshop: "32 activation stacks per species" (§3).
- Main paper: `MixedCanonicalSpecies` uses "4 × N LoRA stacks" with
  N configurable. The "stack" label is used differently.
- **Fix**: pick one. If "32 stacks" in workshop is the same concept as
  `N=32` in `MixedCanonicalSpecies`, document the correspondence.

### L2. Workshop main.tex preamble now requires `natbib`, `cleveref`,
`algorithm`, `algpseudocode`

- Added to main.tex preamble in commit `4fe5ba5` to accommodate the v2
  section commands. Not an error — just worth a note so that any
  future v2→v3 pass is aware of the new dependency set.

### L3. Duplicate labels generate tectonic warnings

- `paper/workshop` tectonic build produces `Object @figure.3 already
  defined` and two `@table.X already defined` warnings. These are
  duplicate `\label{...}` across v2 sections, not caught by the build
  since LaTeX overrides silently.
- **Fix**: grep `^\\label\b` in all six section files, rename
  duplicates, rebuild.

### L4. CHANGELOG [Unreleased] is one commit behind main

- The `paper/` asym-drift systematic finding (commit `3274e99`) isn't
  summarized in `CHANGELOG.md` `[Unreleased]`. Minor; a bullet in
  Added would close the loop.

---

## Not issues (ruled out during this review)

- Main paper's 256-test claim is accurate (5 new drift-splitting tests
  on top of the 256 baseline bring the count to 261, but the main
  paper text still says 256 — close enough; the 261 count is in the
  feb6774 commit message, not repeated in the paper).
- Workshop's `saillant2026nervewml` citation is not yet in its own
  references.bib — MEDIUM bug on the workshop side alone (already
  known: workshop prose cites forward to papers not yet published),
  out of scope for this cross-paper review.

---

## Triage recommendation for tomorrow morning (fresh)

1. Fix **C1** (species order in workshop prose) — 10 minutes, sed-safe,
   unblocks all downstream consistency.
2. Fix **H3** (unify citation set) — 5 minutes of editorial choice.
3. Decide **M1** wording to close the reviewer narrative risk — 15
   minutes of careful prose.
4. Everything else can wait for a v0.12-draft pass or whenever.

Estimated total time: ~30 minutes fresh, vs. 60 minutes at 22:30 tired.

_--- notes generated 2026-04-22 22:35 CEST by late-night critic pass ---_
