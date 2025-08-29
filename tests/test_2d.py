import os
import pytest


try:
    from rdkit import Chem  # noqa: F401
    RDKIT = True
except Exception:  # pragma: no cover - environment w/o RDKit
    RDKIT = False


@pytest.mark.skipif(not RDKIT, reason="RDKit not available")
def test_2d_screening(tmp_path):
    from pharmacophore_screening import PharmacophoreScreener
    pkg_root = os.path.dirname(os.path.dirname(__file__))
    lib = os.path.join(pkg_root, "samples", "compounds_small.csv")

    s = PharmacophoreScreener(mode="2D")
    res = s.screen(query_smiles="CCO", library_csv=lib, similarity_threshold=0.0, n_workers=1)
    df = res.df
    assert "similarity" in df.columns
    assert len(df) > 0

