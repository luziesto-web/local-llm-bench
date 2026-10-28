# Local LLM benchmark on a laptop GPU (8 GB)

An independent benchmark of open-weight models run locally through Ollama on one consumer laptop: which model and
which settings (reasoning mode, temperature, context and reply budget) actually solve small coding tasks, and how
long they take.

- **Ladder of task levels**: Light (L0-L11) -> Medium (M1-M5) -> High (H0-H9) -> qualification -> Mastermind;
  every task has hidden checks run on the files the model wrote (`cases.py`, `light10_cases.py`).
- **Method**: config matrix per model -> screen -> confirm on more seeds -> contest; context budgets calibrated
  from measured token use; hard time limits per episode (High: 12 min); a time-out is a model outcome.
- **Models so far**: Gemma 4 (E2B/E4B), Qwen 3.5 (4B/9B), Qwen3-Coder 30B, Granite 4.2, gpt-oss 20B,
  Nemotron 3.5 Lightning 30B, GLM 4.7 Flash, K2-Horizon 7B.
- **Results**: `runs/<run>/results.jsonl` - one line per episode (model, config, options, task, seed, code_pass,
  time, token use). Full model transcripts are not published.

Updated daily. Last update: 2026-10-28 19:40 UTC.
