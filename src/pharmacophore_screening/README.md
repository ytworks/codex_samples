# pharmacophore_screening

Pharmacophore screening library using RDKit.

- 2D mode: Morgan/ECFP + Tanimoto similarity
- 3D mode: RDKit feature points vs. user-defined pharmacophore points
- Parallel execution via Ray when available; falls back to serial

## Install

Ensure RDKit, pandas, numpy (and optional `ray`) are available in your environment.

## Python API

```python
from pharmacophore_screening import PharmacophoreScreener

# 2D
s2d = PharmacophoreScreener(mode='2D')
res2d = s2d.screen(
    query_smiles="CCO",
    library_csv="samples/compounds_small.csv",
    similarity_threshold=0.3,
    n_workers=1,
)
res2d.to_csv("results_2d.csv")

# 3D
s3d = PharmacophoreScreener(mode='3D')
res3d = s3d.screen(
    pharmacophore_csv="samples/pharmacophore_example.csv",
    library_csv="samples/compounds_small.csv",
    n_conformers=5,
    n_workers=1,
)
res3d.to_csv("results_3d.csv")
```

## CLI

```bash
python -m pharmacophore_screening 2d --query "CCO" --library samples/compounds_small.csv -o results_2d.csv
python -m pharmacophore_screening 3d --ph4 samples/pharmacophore_example.csv --library samples/compounds_small.csv -o results_3d.csv
```

## Notes

- The 3D matching uses a simple nearest-feature assignment under per-feature tolerances and reports RMSD across matched points. It is a pragmatic baseline, not a full pharmacophore alignment.
- Provide a column `smiles` in the input CSV; other columns are preserved and returned with results.
