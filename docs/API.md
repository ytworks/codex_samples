# pharmacophore_screening API

## Classes

- PharmacophoreScreener(mode: '2D' | '3D')
  - screen(...): Run screening and return `ScreeningResult`.
    - 2D args: `query_smiles`, `library_csv`, `similarity_threshold=0.0`, `radius=2`, `n_bits=2048`, `n_workers=1`, `batch_size=None`, `chunksize=None`.
    - 3D args: `pharmacophore_csv`, `library_csv`, `n_conformers=10`, `n_workers=1`, `batch_size=None`, `chunksize=None`, `seed=0`.
  - plot_2d(smiles, out_path, size=(300, 300))
  - write_sdf(smiles, out_path, n_conformers=1, seed=0)

- ScreeningResult
  - df: pandas.DataFrame
  - to_csv(path, index=False)

## Notes

- Input CSVs must include a `smiles` column. All other columns are preserved and returned with additional results columns.
- 2D results include a `similarity` column (float) and are ranked descending by similarity.
- 3D results include `match` (bool), `rmsd` (float), `conformer_id` (int), `details` (list of per-feature matches), and optional `error`.
