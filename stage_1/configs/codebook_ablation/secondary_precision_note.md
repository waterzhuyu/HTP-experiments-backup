# Secondary bootstrap precision amendment

2026-09-22: The original 2,000-draw plus-one bootstrap p-value floor is
1/2001; with 120 Holm comparisons the first-step floor is 120/2001 > 0.05.
Uniformly increase all 15 pairs / 120 comparisons to 20,000 draws, keeping
seed 271828, train-only support, population, metrics, checkpoints and
comparison family fixed. No selective reruns or model retraining.
Old 2,000-draw artifacts remain immutable; use domains_*_002 outputs.
The new floor is 120/20001 = 0.0059997. Finite Monte Carlo uncertainty
remains, especially around the significance threshold. The primary
crossed-seed analysis and equivalence margins are unchanged.
