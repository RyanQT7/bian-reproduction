"""Create leakage-safe BiAn inputs for the ten known incident windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bian.data.real_preprocessing import preprocess_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "real_preprocessing.example.yaml",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    audit = preprocess_dataset(
        raw_root=args.raw_root,
        bundle_root=args.bundle_root,
        output_root=args.output_root,
        config=config,
    )
    print(
        f"Preprocessing: {audit['status']}; cases={audit['evaluated_cases']}; "
        f"candidates={audit['candidate_count']}"
    )
    print(f"Output: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
