# K2 Q5 — calibration plan (run 2026-10-28)

Fixed: tag `hf.co/bartowski/K2-Horizon-7B-GGUF:Q5_K_M`, T1.0, top_p 0.95, no `think` key (template default), q8 KV
cache and flash attention verified and left unchanged, one generation at a time, no automatic context growth.

## Stages (≤ 3–3.75 h of machine time)
| Stage | What | Limit |
|---|---|---|
| 0. Rig (no model) | references and mutants pass/fail as expected; hard episode deadline; nothing keeps generating after a time-out; safe resume | — |
| 1. Speed | contexts 8K/12K/16K/24K, seed 11, 256-token reply, load time, tokens/s, VRAM placement | 12 min for the stage |
| 2. Medium screen | M-C8-P4, M-C12-P4, M-C16-P4, M-C12-P2 × M2, M5, seed 11 | 240 s per episode |
| 3. Medium confirm | one finalist × M1, M3, M4 × seeds 22, 33 | 240 s |
| 4. High screen | H-C12-P8, H-C16-P8, H-C24-P8, H-C16-P4 × H2, H3, seed 11 | 720 s |
| 5. High confirm | one finalist × H4, H8, H5 × seed 22 | 720 s |
| 6. Report | admission: Medium — 2/2 in the screen and ≥5/6 in the confirm; High — 2/2 in the screen, H4 and H8 mandatory; H5 unsolved → label "limited High (H5)" | — |

Finalist: most strict successes → least total time → fewest protocol errors → config id.
No retries on time-out, code failure, parser failure, output cap or OOM; at most one transport retry within the
episode deadline; 2 infrastructure failures in a row → pause.

## Result
Medium 8K/4K (screen 2/2, confirm 5/6); High 12K/8K (screen 2/2, H4 and H8 solved, H5 not → limited High).
Smaller contexts won: more of the model stays on the GPU (≈26 vs 17 tokens/s at 24K). One blocked attempt to write an
extra test file in H8; admitted by the owner.
