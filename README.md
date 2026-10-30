# Local LLM benchmark on a laptop GPU (8 GB)

An independent benchmark of open-weight models run locally through Ollama on one consumer laptop: which model and
which settings (reasoning mode, temperature, context and reply budget) actually solve small coding tasks, and how
long they take.

## The goal: a local assembly line, not one "best model"

On 8 GB of VRAM no single model is both fast and reliable. One big model for every step is too slow; one small
model for every step is too unreliable. So the benchmark is built around **roles**: one prompt goes in, a planner
breaks it into steps, each step is delegated to the cheapest model that does that kind of step reliably, and a
strong model checks the result. The levels below exist to find the right model and settings for each role.

| Role (tag size) | What it does in the line | What matters | Typical size |
|---|---|---|---|
| **SuperLight** (XS) | routing, classification, extracting fields, filtering - one shot, strict machine-readable output | speed, format discipline | 0.8-3B |
| **Light** (S) | small code edits: one module, one trap, with tools | fast and mostly right | 3-4B |
| **Medium** (M) | multi-file fixes, recovering when a tool fails, routine executor work | balance of quality and speed | 7-9B |
| **High** (L) | heavy but quality executors: concurrency, permissions, versioned data, bugs from logs; 12-min cap | correctness first, bounded time | 20-30B (MoE) |
| **Mastermind** (XL) | 1-2 most reliable models: plan, decompose, delegate, **review and verify** the others' work | quality only - time is irrelevant | largest that runs (CPU offload allowed) |

The target line: **one or two reliable Mastermind models** for planning and checking, **heavy but high-quality
High executors**, **steady Medium executors**, and **light, fast Light/SuperLight workers** for the many small
steps - assembled into one cycle driven by a single prompt with efficient delegation. Between High and Mastermind
sits a qualification level that decides who may be trusted with the checker role.

## The final exam for High models (in preparation)

Short coding tasks show who can fix a module. The final exam shows who can **build a whole product**. Every High
configuration gets the same brief: build a real, reusable **team collaboration website from scratch** - a Python
server with a real database, not a mock-up - in ten consecutive stages, each in a fresh conversation with only a
handoff note from its own previous stage. Accounts and sessions, shared workspaces with roles and access
boundaries, work items with versioning and conflicting edits, comments, invitations, search and filters, a summary
view, an accessible UI and demo data.

Each model family builds the same product under its own theme, so the sites cannot be copied from each other:

| Family | Site | Theme |
|---|---|---|
| gpt-oss 20B | Fieldnote | editorial desk |
| Qwen3-Coder 30B | Launchboard | product team portal |
| Nemotron 3.5 Lightning 30B | Labbook | research team journal |
| GLM 4.7 Flash | Craftdesk | design studio |
| Qwen 3.5 9B | Studyflow | study group |
| K2-Horizon 7B | Eventroom | events team |

Models work inside an isolated sandbox (one locked-down user per entry, no internet, no access to other entries or
to the checker). An independent acceptance suite - not shown to the models - scores each site; a stage that breaks
the foundation blocks the stages built on it. The exact tasks and checks stay private until the exam is run.

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
| k2-horizon-7b-q5 | 7/8 (88%) | 1.4 | 39/45 (87%) | 2.7 | 32/45 (71%) | 2.4 |
| qwen3.5-9b | — | — | 35/60 (58%) | 1.5 | 24/33 (73%) | 4.6 |
| gemma4-e2b | 25/92 (27%) | 0.2 | — | — | — | — |
| granite4.2-8b | — | — | 50/88 (57%) | 1.0 | — | — |
| gpt-oss-20b | 3/8 (38%) | 1.6 | 3/8 (38%) | 1.6 | 37/47 (79%) | 3.8 |
| nemotron-3.5-lightning-30b | — | — | — | — | 34/58 (59%) | 5.7 |
| glm-4.7-flash | — | — | — | — | 19/39 (49%) | 2.4 |
| qwen3-coder-30b | — | — | — | — | 14/29 (48%) | 2.1 |
| gemma4-26b | — | — | — | — | 15/18 (83%) | 5.1 |
| qwen3.5-0.8b | 3/17 (18%) | 0.2 | — | — | — | — |
| k2-horizon-7b-q4 | — | — | 5/13 (38%) | 0.7 | — | — |
| gemma4-12b | — | — | — | — | 3/6 (50%) | 12.0 |
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
  time, token use). The dashboard also provides redacted task conditions and clipped model turns, including
  reasoning and tool calls. Repeated request histories and local paths are not published.

**How this was built**: the code was written by AI coding agents (Claude, ChatGPT/Codex) under my direction; the
task ladder, the method, the decisions and the conclusions are mine.

Updated hourly while runs are active. Last update: 2026-10-30 13:00 UTC.
