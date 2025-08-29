import os
import pytest


try:
    from rdkit import Chem  # noqa: F401
    RDKIT = True
except Exception:  # pragma: no cover - environment w/o RDKit
    RDKIT = False


@pytest.mark.skipif(not RDKIT, reason="RDKit not available")
def test_3d_screening(tmp_path):
    from pharmacophore_screening import PharmacophoreScreener
    pkg_root = os.path.dirname(os.path.dirname(__file__))
    lib = os.path.join(pkg_root, "samples", "compounds_small.csv")
    ph4 = os.path.join(pkg_root, "samples", "pharmacophore_example.csv")

    s = PharmacophoreScreener(mode="3D")
    res = s.screen(pharmacophore_csv=ph4, library_csv=lib, n_conformers=2, n_workers=1)
    df = res.df
    assert "match" in df.columns
    # We don't assert hits because the sample pharmacophore may not match
    assert len(df) > 0

