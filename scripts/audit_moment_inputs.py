#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from bian.detection.guard import append_audit
from bian.detection.preprocessing import audit_inputs

p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--output",required=True)
a=p.parse_args(); result=audit_inputs(a.input); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
append_audit(out.parent/"truth_access_audit.jsonl","audit_input",str(Path(a.input).resolve()))

