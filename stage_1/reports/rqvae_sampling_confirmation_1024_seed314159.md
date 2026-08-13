# Released HTP Token Sampling Pilot

## Chengdu

- Sample: 1024 held-out trajectories; native cadence 10 s; one token spans roughly 80 s before receptive-field effects.
- 2× downsampling: layer-1 agreement 0.177, tuple agreement 0.000, ARI 0.131.
- 3× downsampling: layer-1 agreement 0.078, tuple agreement 0.000, ARI 0.058.
- 20% missing / 10 m noise layer-1 agreement: 0.374 / 0.923.
- Speed-state probe macro-F1: 0.498 vs majority 0.177; token coverage 1.000.
- NMI with speed/road/grid: 0.202 / 0.495 / 0.035.
- Same-motion 2× vs same-road hard-negative histogram similarity: 0.312 vs 0.3411404695698298.
- Uncertainty: unavailable in the deterministic released encoder.

## Porto

- Sample: 1024 held-out trajectories; native cadence 15 s; one token spans roughly 120 s before receptive-field effects.
- 2× downsampling: layer-1 agreement 0.122, tuple agreement 0.000, ARI 0.106.
- 3× downsampling: layer-1 agreement 0.086, tuple agreement 0.000, ARI 0.040.
- 20% missing / 10 m noise layer-1 agreement: 0.205 / 0.848.
- Speed-state probe macro-F1: 0.448 vs majority 0.148; token coverage 1.000.
- NMI with speed/road/grid: 0.123 / 0.399 / 0.054.
- Same-motion 2× vs same-road hard-negative histogram similarity: 0.220 vs 0.23918554026333336.
- Uncertainty: unavailable in the deterministic released encoder.

## Pilot Verdict

The released HTP tokens are a valid non-collapsed reconstruction baseline but do not pass the Stage 1 semantic-primitive gate. Sampling consistency is below the provisional 0.70 layer-1 threshold in at least one city, and the model has no uncertainty mechanism. These results support testing P1–P4, not interpreting the released tokens as motion primitives.
