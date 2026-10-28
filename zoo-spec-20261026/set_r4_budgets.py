"""Round 4 budgets (owner 2026-10-28): Qwen 9B x1.5 of r3 (do not choke its reasoning), gpt-oss and GLM raised
against the HTTP 500s; the other models wait for the r3 report (status hold)."""
import json
from pathlib import Path
p = Path(__file__).with_name("calibration.json")
c = json.loads(p.read_text(encoding="utf-8"))
Path(__file__).with_name("calibration.r3.json").write_text(json.dumps(c, indent=2), encoding="utf-8")
r4 = {"qwen3.5-9b": (110592, 55296), "gpt-oss-20b": (49152, 24576), "glm-4.7-flash": (65536, 8192)}
for k, v in c.items():
    if k in r4:
        v.update(num_ctx=r4[k][0], num_predict=r4[k][1], status="proposed-r4")
    elif v["status"] != "validated":
        v["status"] = "hold"
p.write_text(json.dumps(c, indent=2), encoding="utf-8")
print({k: (v["num_ctx"], v["num_predict"], v["status"]) for k, v in c.items()})
