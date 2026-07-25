# Configuration

Configuration values carry one of three provenance labels:

- `paper-reported`
- `implementation-choice`
- `unknown/not-reported`

The project uses JSON-compatible YAML files so the first phase can parse them
with Python's standard-library `json` module. Local absolute paths belong in
ignored `configs/paths.local.yaml`, never in committed examples.
