# codex_samples

This repository contains sample projects. The `pharmacophore_screening` Python package implements 2D/3D pharmacophore screening with RDKit and optional Ray parallelism.

- Package: `src/pharmacophore_screening/`
- Quickstart: see `src/pharmacophore_screening/README.md`
- Samples: `samples/`

Environment setup:

```bash
# Recommended (conda/mamba)
make setup  # creates/updates conda env 'phscreen' or a local venv

# Alternatively (pip/venv)
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
```

Run CLI examples:

```bash
python -m pharmacophore_screening 2d --query "CCO" --library samples/compounds_small.csv -o results_2d.csv
python -m pharmacophore_screening 3d --ph4 samples/pharmacophore_example.csv --library samples/compounds_small.csv -o results_3d.csv
```
