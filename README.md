# Local LLM benchmark on a laptop GPU (8 GB)

An independent benchmark of open-weight models run locally through Ollama on one consumer laptop: which model and
which settings (reasoning mode, temperature, context and reply budget) actually solve small coding tasks, and how
long they take.

## Findings so far

1. **Light tasks**: Gemma 4 E4B and Qwen 3.5 4B tie (9/12 correct code) in the config matrix; for every model the
   vendor settings with reasoning on beat "no reasoning" and a "check yourself" system hint.
2. **Medium tasks**: K2-Horizon 7B Q5 solved 13/15, Granite 4.2 8B and Qwen 3.5 9B 11/15 - Granite the fastest.
3. **The harness can be the bug**: forcing JSON-formatted answers failed gpt-oss for reasons unrelated to coding;
   native tool calls fixed it. JSON is kept only where machine-readable output is the point of the task.
4. **A 12-minute cap per High episode loses nothing**: every solved High episode finished within ~10 minutes; the
   failures ran 17-60 minutes. The cap cut 4-5 hours of a 14-hour night.
5. **gpt-oss "hangs" are reasoning, not stalls**: at high effort it can think for 90+ s about copying a file that low
   effort writes in 5 s. Ollama's HTTP 500 is an intermittent "error parsing tool call", not memory. The harness
   was also dropping gpt-oss reasoning between tool calls (fixed).
6. **Less context can be better on 8 GB**: K2 Q5 did best at 8K (Medium) and 12K (High) - more of the model stays
   in VRAM (26 vs 17 tokens/s at 24K).
7. **Check the task before blaming the model**: a High task solved 0/29 times partly because its spec left the
   lease-expiry boundary open; the spec now states it and the checks report independent groups.

## Results by model and level

All published episodes (every config and seed, so not the best-config score). Solved = hidden checks passed on
the files the model wrote; time = wall-clock minutes per episode.

| Model | Light solved | Light median min | Medium solved | Medium median min | High solved | High median min |
|---|---|---|---|---|---|---|
| gemma4-e4b | 94/160 (59%) | 0.2 | — | — | — | — |
| qwen3.5-4b | 45/82 (55%) | 0.3 | 16/40 (40%) | 0.5 | — | — |
| granite4.2-3b | 64/103 (62%) | 0.5 | — | — | — | — |
| gemma4-e2b | 25/92 (27%) | 0.2 | — | — | — | — |
| granite4.2-8b | — | — | 50/88 (57%) | 1.0 | — | — |
| qwen3.5-9b | — | — | 35/60 (58%) | 1.5 | 10/19 (53%) | 3.6 |
| k2-horizon-7b-q5 | — | — | 36/41 (88%) | 2.7 | 17/21 (81%) | 2.6 |
| nemotron-3.5-lightning-30b | — | — | — | — | 16/25 (64%) | 4.1 |
| gpt-oss-20b | — | — | 0/4 (0%) | 1.6 | 15/17 (88%) | 4.5 |
| glm-4.7-flash | — | — | — | — | 10/21 (48%) | 2.2 |
| qwen3-coder-30b | — | — | — | — | 10/20 (50%) | 1.9 |
| qwen3.5-0.8b | 3/17 (18%) | 0.2 | — | — | — | — |
| k2-horizon-7b-q4 | — | — | 5/13 (38%) | 0.7 | — | — |
| bonsai-2-9b | — | — | 2/5 (40%) | 1.0 | — | — |
| k2-horizon-7b-q8 | — | — | 4/5 (80%) | 2.2 | — | — |
| devstral-24b | — | — | — | — | 1/4 (25%) | 23.1 |

- **Ladder of task levels**: Light (L0-L11) -> Medium (M1-M5) -> High (H0-H9) -> qualification -> Mastermind;
  every task has hidden checks run on the files the model wrote (`cases.py`, `light10_cases.py`).
- **Method**: config matrix per model -> screen -> confirm on more seeds -> contest; context budgets calibrated
  from measured token use; hard time limits per episode (High: 12 min); a time-out is a model outcome.
- **Models so far**: Gemma 4 (E2B/E4B), Qwen 3.5 (4B/9B), Qwen3-Coder 30B, Granite 4.2, gpt-oss 20B,
  Nemotron 3.5 Lightning 30B, GLM 4.7 Flash, K2-Horizon 7B.
- **Results**: `runs/<run>/results.jsonl` - one line per episode (model, config, options, task, seed, code_pass,
  time, token use). Full model transcripts are not published.

Updated daily. Last update: 2026-10-28 19:42 UTC.
