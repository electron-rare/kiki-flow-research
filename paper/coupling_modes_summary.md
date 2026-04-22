# Coupling-mode ablation summary

Sweep: 45 cells (3 alpha x 3 beta x 5 modes), 5 seeds each.
Runtime: 220.3 s.  Best (lowest H) dell-full cell: alpha=10.0, beta=1.0.

## Mean per-species entropy (bits) at best dell-full cell

| mode | mean H | std |
| --- | --- | --- |
| separable | 1.9612 | 0.0201 |
| symmetric-dell | 1.9675 | 0.0238 |
| dell-full | 1.9675 | 0.0238 |
| symmetric-levelt | 1.9687 | 0.0240 |
| levelt-full | 1.9687 | 0.0240 |

## Comparisons

- separable vs dell-full: Delta H = -0.0063 bits (-0.32%).
- separable vs levelt-full: Delta H = -0.0076 bits.
- symmetric-dell vs dell-full: Delta H = +0.0000 bits (tests whether J_asym drift matters in the solver path).
- symmetric-levelt vs levelt-full: Delta H = +0.0000 bits.
- dell-full vs levelt-full: Delta H = -0.0012 bits (tests whether coupling topology matters).

## Notes

- `T2FreeEnergy.value()` in `paper_f.py` routes the scalar coupling energy through `J_sym` only (the antisymmetric contraction vanishes identically), so the JKO gradient path sees `J_sym` regardless of whether `J_asym` is zeroed. `symmetric-*` and `*-full` modes are therefore expected to coincide to within numerical noise on this solver -- the ablation primarily isolates (separable vs coupled) and (Dell vs Levelt topology).
- `max_entropy = log2(n_species * grid) = 6.0000` bits per task spec; per-species entropy is bounded by log2(grid) = 4.0000 bits.
