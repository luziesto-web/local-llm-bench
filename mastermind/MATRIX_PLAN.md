# Mastermind (XL): per-model run plan — final specs (2026-10-28)

Role: planner and "senior engineer". **Time is irrelevant — only quality and finished tasks count.**
Tasks A–E: A unfamiliar library from its docs, B change a live service without breaking it, C how to test untested
code (mutants), D analyse someone else's failure, E honesty under pressure.
All values below were checked against the model creators' cards on 2026-10-28 (sources at the end).

## Rules
1. Creator configs (V) first — exactly as recommended. Then our own (O).
2. 8 GB GPU: CPU offload is allowed; context is not cut for speed.
3. One model = one run `runs/mastermind-<model>-<date>`.
4. Ranking: solved tasks only (`mm_suite check`) and honesty (E; truthful `checks_run`). Speed is for reference.

## Common conditions
- Ollama with `OLLAMA_KV_CACHE_TYPE=q8_0` + `OLLAMA_FLASH_ATTENTION=1` (recorded in the report).
- No time limit. `done_reason=length` → flagged and the episode re-run once, not a model failure. HTTP 400/500 is not
  a model result.
- Identical for all: task text (`mm_suite prepare`), harness, LLM-free checking.
- Order per model: all V configs on A–E (seed 42) → all O configs on A–E (seed 42) → the two best configs on A–E with
  seeds 314159 and 271828 (confirmation).
- **Context rule.** If the creator recommends a context — use it. If only a maximum is given — use 131072 (or the
  model maximum if smaller), but never less than "recommended reply + 32K". If the model does not load with it
  (out of memory at load, 32 GB RAM + 8 GB VRAM) — step down 131072 → 98304 → 65536 and set
  `num_predict = num_ctx − 32768`; recorded as a hardware limit, not a model limit.
- "Hint" (O): one line at the end of the system message: `Before writing ANSWER.json, run your own checks and report
  only checks you actually ran.`

## 1. gpt-oss:20b — tag `gpt-oss:20b`
Creator: temperature 1.0, top_p 1.0; reasoning effort low/medium/high (default low; high = "deep and detailed
analysis"), in Ollama `think: "low"|"medium"|"high"`. No recommended context or reply length.
| id | think | options |
|---|---|---|
| V1 | `"high"` | temperature 1.0, top_p 1.0, num_ctx 131072, num_predict 65536 |
| V2 | `"medium"` | same |
| O1 | `"high"` | same + hint |
| O2 | `"high"` | temperature 0.6, top_p 1.0, num_ctx 131072, num_predict 65536 |
| O3 | `"high"` | as V1 but num_ctx 32768, num_predict 24576 (what a cut costs) |

## 2. nemotron-3.5-lightning:30b — `use_mmap: true`
Creator: temperature 1.0, top_p 0.95 (all modes, tools included); reasoning on by default, off via
`enable_thinking=False` (Ollama `think: false`); no reasoning budget; 256K context on one H100; `max_tokens=16000` in
all examples. Code agents: `force_nonempty_content` (not available in Ollama — noted).
| id | think | options |
|---|---|---|
| V1 | `true` | temperature 1.0, top_p 0.95, num_ctx 262144, num_predict 16000, use_mmap |
| O1 | `true` | same, num_predict 32768 |
| O2 | `false` | as V1 (control without reasoning) |
| O3 | `true` | as V1 + hint |

## 3. glm-4.7-flash — tag `glm-4.7-flash:latest`, `use_mmap: true`
Creator: default ("most tasks") temperature 1.0, top_p 0.95, max new tokens 131072; Terminal Bench / SWE-bench
Verified: temperature 0.7, top_p 1.0, max new tokens 16384; multi-turn agents use "Preserved Thinking" (past
reasoning returned to the history). No context recommendation.
| id | think | options |
|---|---|---|
| V1 | `true` | temperature 1.0, top_p 0.95, num_ctx 163840, num_predict 131072, use_mmap |
| V2 | `true` | temperature 0.7, top_p 1.0, num_ctx 131072, num_predict 16384, use_mmap, preserved thinking* |
| O1 | `true` | as V2 + hint |
| O2 | `false` | as V2 (control) |
\* preserved thinking: the harness returns the `thinking` field of past replies to the message history; if the
harness cannot, V2 runs without it, flagged as a deviation from the creator.

## 4. qwen3-coder:30b — `use_mmap: true`
Creator: temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.05; reply 65536 "for most queries"; native
context 262144 (32768 suggested when memory is short); non-thinking only (`think` not sent).
| id | think | options |
|---|---|---|
| V1 | — | temperature 0.7, top_p 0.8, top_k 20, repeat_penalty 1.05, num_ctx 131072, num_predict 65536, use_mmap |
| O1 | — | same + hint |
| O2 | — | temperature 0.3, rest as V1 |

## 5. qwen3.5:9b
Creator: reasoning on by default (`think: false` to disable); context "at least 128K to keep reasoning quality";
reply 32768 usually, **81920 for complex programming**. Presets (all min_p 0, repetition 1.0): reasoning/general —
T1.0, top_p 0.95, top_k 20, presence 1.5; reasoning/precise code — T0.6, top_p 0.95, top_k 20, presence 0.0.
| id | think | options |
|---|---|---|
| V1 | `true` | temperature 0.6, top_p 0.95, top_k 20, min_p 0, presence_penalty 0.0, repeat_penalty 1.0, num_ctx 131072, num_predict 81920 |
| V2 | `true` | temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 1.5, repeat_penalty 1.0, num_ctx 131072, num_predict 81920 |
| O1 | `true` | as V1 + hint |
| O2 | `true` | as V1 but num_ctx 32768, num_predict 24576 |

## 6. granite4.2:8b
Creator: temperature 1.0, top_p 0.95 "for all tasks and backends"; full reasoning on by default, `low_effort`
available; reply 8192 in reasoning mode "(increase for complex reasoning)"; native context 128K.
| id | think | options |
|---|---|---|
| V1 | `true` | temperature 1.0, top_p 0.95, num_ctx 131072, num_predict 8192 |
| V2 | `true` | same, num_predict 32768 (the creator allows raising it) |
| O1 | `true` | temperature 0.7, top_p 0.9, top_k 50 (Ollama tag defaults), num_ctx 131072, num_predict 32768 |
| O2 | `true` | as V2 + hint |

## 7. K2-Horizon 7B — tags `hf.co/bartowski/K2-Horizon-7B-GGUF:Q5_K_M` and `ingloryon/k2-horizon:7b-q4_K_M`
Creator (IFM/K2-Horizon-7B): temperature 1.0, top_p 0.95; reasoning_effort high (medium/low "not recommended except
for research"); reply "at least 32768 so reasoning is not cut"; native context 524288.
The Q5 tag answers HTTP 400 to the `think` parameter → reasoning by template (no parameter); the Q4 tag accepts
`think: "high"`.
| id | tag | think | options |
|---|---|---|---|
| V1 | Q5 | — (template) | temperature 1.0, top_p 0.95, num_ctx 131072, num_predict 32768 |
| V2 | Q4 | `"high"` | same |
| O1 | Q5 | — | as V1 + hint |
| O2 | Q5 | — | same, num_predict 65536 |

## Size
7 models, 28 configs (12 V + 16 O). Screen: 28 × 5 tasks = 140 episodes. Confirmation: 7 × 2 configs × 5 tasks ×
2 seeds = 140. Days of laptop time; a run stops and resumes without losses.

## Why it might not work
1. JSON-turn harnesses can hurt reasoning models — format errors are counted separately.
2. 131K–262K contexts may not load into 40 GB of memory — then step down by the context rule.
3. Long reasoning can eat the reply limit — flagged and re-run, not a failure.
4. Three seeds per task — a one-episode difference is noise.

## Sources (read 2026-10-28)
- gpt-oss: https://github.com/openai/gpt-oss, https://huggingface.co/openai/gpt-oss-20b
- Nemotron 3.5 Lightning: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
- GLM-4.7-Flash: https://huggingface.co/zai-org/GLM-4.7-Flash
- Qwen3-Coder: https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct
- Qwen3.5 9B: https://huggingface.co/Qwen/Qwen3.5-9B
- Granite 4.2 8B: https://huggingface.co/ibm-granite/granite-4.2-8b
- K2-Horizon 7B: https://huggingface.co/IFM/K2-Horizon-7B
