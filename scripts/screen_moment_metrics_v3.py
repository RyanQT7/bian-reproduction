#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from bian.detection.screening_v3 import screen
p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--output",required=True)
a=p.parse_args(); print(json.dumps(screen(Path(a.input),Path(a.output),512),indent=2))
