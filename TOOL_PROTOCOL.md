# Answer format in the tests (decision 2026-10-28)

"Keep JSON where it is justified; rework the tasks where it is not."

Why: in Medium, gpt-oss answered with native tool calls (`repo_browser.list_files`) and empty text — 32 turns of
"JSONDecodeError", it never reached the task; forcing `format=json` did not change that. K2 used to trip on "Reply
must be one JSON object". The test was measuring "does the model keep our format" mixed with "does it solve the
task", and did not match how models are used in practice (OpenCode uses native tools).

## JSON is justified — kept
| test | why |
|---|---|
| SuperLight (S1–S6) | the JSON answer is the task itself (short structured answers) |
| Site pipeline: fact extraction, plan, assessments | a machine-readable answer is needed; now short codes decoded by a parser |

## JSON is not justified — moved to Ollama native tools
| test | status |
|---|---|
| Light / Medium / High (`run_light.py` via `zoo_night.py`) | **done**: `tool_mode: "native"` — actions are Ollama tools, results come back as `tool` messages. Default stays `json` so old runs remain comparable. |
| All agent tests, both modes | **done**: a native tool call is always accepted and turned into our action; format errors are counted separately (`format_errors`, `native_tool_calls` in episode.json) |
| C++ (`cpp_bench/cpp_bench.py`, own loop, `format=json`) | **to do**: move to the same mode |
| Mastermind A–E | harness built on native tools from the start |

## What this means for old results
Failures caused by the format are not task failures. Re-run in `native` mode: gpt-oss (Medium/High), K2 (episodes
with format errors), the rest on request. Harness tests: `test_light_runner.py`.
