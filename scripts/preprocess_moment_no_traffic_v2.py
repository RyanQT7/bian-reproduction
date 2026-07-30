#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from bian.detection.no_traffic_v2 import preprocess

p=argparse.ArgumentParser()
p.add_argument("--input",required=True)
p.add_argument("--output",required=True)
a=p.parse_args()
print(json.dumps(preprocess(Path(a.input),Path(a.output)),indent=2))
