# BiAn Engineering Reproduction

This project reproduces the method structure and engineering flow described in
“Towards LLM-Based Failure Localization in Production-Scale Networks”.

The current phase uses synthetic examples and a deterministic mock model. Every
generated result is labeled:

> 流程验证结果 / Pipeline Validation Result

Mock results are not paper reproduction accuracy and must not be compared with
the paper's production results.

## Parameter provenance

- `paper-reported`: explicitly stated in the paper.
- `implementation-choice`: selected for this engineering implementation.
- `unknown/not-reported`: required detail not disclosed by the paper.

The default Rank of Ranks rounds (`3`) and entropy threshold (`0.75`) are
paper-reported. Raw entropy as the default mode and the Stage 2 device Top-p
threshold are implementation choices. Generation temperature, generation
Top-p, maximum tokens, and the data-source-to-anomaly mapping remain
configurable.

## Current scope

The first phase covers schemas, validation, deterministic mock data/model
backends, topology reduction, timeline construction, candidate filtering,
entropy early stop, Rank of Ranks, the Hot Device baseline, Top-k metrics, a
mock pipeline, standard-library tests, logging, and reproducible run artifacts.

It does not load the local 7B or 32B models, train any model, use CUDA, process
real incidents, or claim the paper's accuracy.

## Safety boundaries

- `paper/paper.pdf` is read-only and excluded from Git.
- `/home/xieqitong/pretrained_models` is external and read-only.
- Generated artifacts go to `outputs/`; command logs go to `logs/`.
- Real datasets, local paths, secrets, weights, logs, and outputs are ignored.

See `configs/README.md` for configuration provenance and
`data/real/README_expected_schema.md` for the future real-data contract.

## Run the first-phase validation

No third-party dependency is required:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 -m scripts.run_mock \
  --config configs/experiments/mock_pipeline.yaml \
  2>&1 | tee "logs/mock_run_$(date +%Y%m%d_%H%M%S).log"
```

Each run creates `outputs/mock_run_YYYYMMDD_HHMMSS/` containing:

- `summary.md`
- `metrics.json`
- `predictions.jsonl`
- `resolved_config.yaml`
- `environment.txt`
- `run.log`
- `tests.log`
- `changed_files.txt`

Outputs and logs are deliberately ignored by Git. The committed mock generator
uses only synthetic device names and synthetic events. Re-running with the same
seed produces identical incidents, scores, rankings, and metrics.

## Implementation notes

- The topology method uses deterministic shortest paths and omits a candidate
  pair path when another candidate lies inside that path. The paper's
  non-candidate device-group aggregation is reserved for a future extension.
- Equal event timestamps receive deterministic microsecond offsets so the
  materialized global timeline is strictly ordered.
- Top-p is applied after softmax, following the paper's description. Its
  threshold is an `implementation-choice`.
- Raw and normalized entropy are supported; raw entropy is the default
  `implementation-choice`. The threshold `0.75` is `paper-reported`, while the
  paper does not report whether its entropy was normalized.
- The mock score formula is not a paper algorithm. It is a deterministic test
  double used only to exercise orchestration and validation.
