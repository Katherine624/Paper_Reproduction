"""Print current progress without importing any model or using GPU memory."""
import json
from pathlib import Path
ROOT=Path("/root/gpufree-data/openvla-repro")
result={}
stage=ROOT/"results/pipeline_stage.txt"
if stage.exists():result["stage"]=stage.read_text().strip()
runs=list((ROOT/"results/eval-spatial-100").glob("*.jsonl"))
if runs:
    run=max(runs,key=lambda x:x.stat().st_mtime)
    rows=[json.loads(x) for x in run.read_text().splitlines() if x.strip()]
    result["evaluation"]={"episodes":len(rows),"successes":sum(x["success"] for x in rows),"errors":sum(x["error"] is not None for x in rows)}
for name in ["metrics","resume_metrics"]:
    path=ROOT/f"results/train-smoke/{name}.jsonl"
    if path.exists():
        lines=path.read_text().splitlines()
        if lines:result[name]=json.loads(lines[-1])
for name in ["pipeline_complete","pipeline_failure"]:
    path=ROOT/f"results/{name}.txt"
    if path.exists():result[name]=path.read_text().strip()
print(json.dumps(result,indent=2))
