#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from bian.detection.guard import FORBIDDEN_NAME, sha256, append_audit
from bian.detection.scoring import score
p=argparse.ArgumentParser(); p.add_argument("--predictions",required=True); p.add_argument("--truth",required=True)
p.add_argument("--frozen-sha",required=True); p.add_argument("--output",required=True)
a=p.parse_args(); pred=Path(a.predictions)
expected=Path(a.frozen_sha).read_text().strip().split()[0]
if sha256(pred)!=expected: raise SystemExit("prediction SHA does not match frozen artifact")
if FORBIDDEN_NAME not in Path(a.truth).resolve().parts: raise SystemExit("unexpected truth directory")
def load(path):
 with Path(path).open(encoding="utf-8") as f: return [json.loads(x) for x in f if x.strip()]
result=score(load(pred),load(a.truth),33); Path(a.output).write_text(json.dumps(result,indent=2),encoding="utf-8")
append_audit(Path(a.output).parent/"truth_access_audit.jsonl","score_truth_after_freeze",str(Path(a.truth).resolve()))
print(json.dumps({k:v for k,v in result.items() if not isinstance(v,list)},indent=2))

