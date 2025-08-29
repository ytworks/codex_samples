import importlib


def test_import_package():
    mod = importlib.import_module("pharmacophore_screening")
    assert hasattr(mod, "PharmacophoreScreener")

