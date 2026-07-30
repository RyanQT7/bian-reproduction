#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from bian.detection.guard import append_audit
from bian.detection.preprocessing import preprocess

p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--output",required=True)
p.add_argument("--manifest",required=True); p.add_argument("--start"); p.add_argument("--end")
a=p.parse_args(); manifest=json.loads(Path(a.manifest).read_text())
print(json.dumps(preprocess(a.input,a.output,manifest,a.start,a.end),indent=2))
append_audit(Path(a.output)/"truth_access_audit.jsonl","preprocess_input",str(Path(a.input).resolve()))

