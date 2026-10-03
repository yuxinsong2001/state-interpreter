# Reproducibility

## Publicly runnable scope

The following checks require no XJTU-SY data or private checkpoint:

```bash
python -m pip install -e .[dev]
python examples/minimal_interpreter_demo.py
python -m pytest -q
```

The minimal example uses deterministic synthetic embeddings to demonstrate only the interpreter API and lifecycle. It is not experimental evidence.

## Environment

- Python: `>=3.10`
- Core dependencies: NumPy, SciPy and PyTorch as declared in `pyproject.toml`
- Development dependency: pytest
- Optional analysis dependencies: matplotlib and scikit-learn
- Minimal example: CPU only
- Historical encoder training: PyTorch CPU or GPU; runtime depends on experiment and hardware

The project intentionally uses version ranges rather than claiming a single fully frozen environment. Historical reports record seeds and key configuration values, but exact reproduction can also depend on PyTorch, hardware and local dataset files.

## Dataset layout

The raw XJTU-SY dataset is not distributed by this repository. Scripts expect a user-provided root containing condition and bearing directories. Use a local path at execution time; do not commit it to configuration files.

Example:

```powershell
python scripts/smoke_test_xjtu_sy.py `
  --root "<DATA_ROOT>" `
  --condition 35Hz12kN `
  --bearing Bearing1_1 `
  --limit 2
```

## Experiment records

- `configs/`: experimental decisions and locked protocol parameters.
- `records/`: completion and access state kept separate from configuration.
- `results/`: compact reports, metrics and figures suitable for review.
- `runs/`: local heavy artifacts; ignored by Git.

The most important public evidence is indexed in [`experiment_index.md`](experiment_index.md).

## Reproduction limitations

The complete historical training and blind-inference runs cannot be reproduced from a fresh clone alone because raw vibration data and large checkpoints are intentionally excluded. Where available, the repository retains configuration files, result summaries and hashes so that local artifacts can be checked without publishing them.

Accordingly, this repository claims:

- reproducible package tests and minimal API example;
- traceable public summaries for selected experiments;
- not one-command full reproduction of every archived experiment.

