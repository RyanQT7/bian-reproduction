#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path
from bian.detection.moment import detect
p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--input",required=True)
p.add_argument("--output",required=True); p.add_argument("--smoke-end")
a=p.parse_args(); cfg=json.loads(Path(a.config).read_text())
if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"0","1","4","5","6"}: raise SystemExit("select one allowed GPU")
print(json.dumps(detect(a.input,a.output,cfg,"cuda:0",a.smoke_end),indent=2))

