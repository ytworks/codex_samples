from __future__ import annotations

import math
import os
import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Dict, Any, Tuple

import pandas as pd

try:  # Optional at import-time: delay heavy modules until needed
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors, DataStructs
    from rdkit.Chem import rdMolAlign
    from rdkit.Chem import ChemicalFeatures
    from rdkit import RDConfig
    _RDKit_AVAILABLE = True
except Exception:  # pragma: no cover - environment without RDKit
    _RDKit_AVAILABLE = False

try:
    import ray  # type: ignore
    _RAY_AVAILABLE = True
except Exception:  # pragma: no cover
    _RAY_AVAILABLE = False


logger = logging.getLogger("pharmacophore_screening")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


@dataclass
class ScreeningResult:
    df: pd.DataFrame

    def to_csv(self, path: str, index: bool = False) -> None:
        self.df.to_csv(path, index=index)


def _ensure_rdkit() -> None:
    if not _RDKit_AVAILABLE:
        raise ImportError("RDKit is required for this functionality but is not available.")


def _init_ray_if_needed(n_workers: int) -> bool:
    if n_workers <= 1:
        return False
    if not _RAY_AVAILABLE:
        logger.warning("Ray not available; falling back to serial execution.")
        return False
    if ray.is_initialized():
        return True
    try:
        ray.init(ignore_reinit_error=True, include_dashboard=False, logging_level=logging.ERROR)
        return True
    except Exception as e:  # pragma: no cover
        logger.warning(f"Failed to initialize Ray: {e}. Falling back to serial execution.")
        return False


def _auto_batch_size(total_rows: int, n_workers: int) -> int:
    # Aim for ~2000 rows per task, scaled by workers, bounded
    target = max(500, min(5000, total_rows // max(1, (n_workers * 4))))
    return max(500, target)


def _chunk_indices(n: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    i = 0
    while i < n:
        j = min(n, i + batch_size)
        yield i, j
        i = j


def _morgan_fp_from_smiles(smiles: str, radius: int = 2, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def _tanimoto(fp1, fp2) -> float:
    return float(DataStructs.TanimotoSimilarity(fp1, fp2))


def _compute_similarity_batch(query_smiles: str, batch_df: pd.DataFrame, radius: int, n_bits: int) -> pd.DataFrame:
    _ensure_rdkit()
    query_fp = _morgan_fp_from_smiles(query_smiles, radius=radius, n_bits=n_bits)
    if query_fp is None:
        raise ValueError("Invalid query SMILES.")
    sims: List[Optional[float]] = []
    for s in batch_df["smiles"].astype(str).tolist():
        try:
            fp = _morgan_fp_from_smiles(s, radius=radius, n_bits=n_bits)
            if fp is None:
                sims.append(None)
            else:
                sims.append(_tanimoto(query_fp, fp))
        except Exception:
            sims.append(None)
    out = batch_df.copy()
    out["similarity"] = sims
    return out


def _load_feat_factory():
    # Load default RDKit pharmacophore features
    fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    return ChemicalFeatures.BuildFeatureFactory(fdef_name)


def _embed_and_optimize_conformers(mol: "Chem.Mol", n_conformers: int = 10, seed: int = 0) -> List[int]:
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
    # Optimize
    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0)
    except Exception:
        try:
            AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=0)
        except Exception:
            pass
    return list(conf_ids)


def _mol_features_3d(mol: "Chem.Mol", conf_id: Optional[int] = None) -> List[Dict[str, Any]]:
    ff = _load_feat_factory()
    feats = ff.GetFeaturesForMol(mol)
    out: List[Dict[str, Any]] = []
    for f in feats:
        # Use 3D centroid of feature atoms
        atom_ids = f.GetAtomIds()
        # Some features might not map to atoms (rare); skip
        if not atom_ids:
            continue
        # Use active conformer coordinates; caller should set conf
        try:
            conf = mol.GetConformer(conf_id) if conf_id is not None else mol.GetConformer()
            xs, ys, zs = [], [], []
            for a in atom_ids:
                pos = conf.GetAtomPosition(a)
                xs.append(pos.x); ys.append(pos.y); zs.append(pos.z)
            x = sum(xs) / len(xs); y = sum(ys) / len(ys); z = sum(zs) / len(zs)
            out.append({
                "type": f.GetFamily(),
                "x": x, "y": y, "z": z,
                "atom_ids": list(atom_ids)
            })
        except Exception:
            continue
    return out


def _parse_pharmacophore_csv(path: str, default_tol: float = 1.0) -> List[Dict[str, Any]]:
    df = pd.read_csv(path)
    req = {"feature_type", "x", "y", "z"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Pharmacophore CSV missing columns: {sorted(missing)}")
    feats = []
    for _, row in df.iterrows():
        feats.append({
            "feature_type": str(row["feature_type"]).strip(),
            "x": float(row["x"]),
            "y": float(row["y"]),
            "z": float(row["z"]),
            "tolerance": float(row.get("tolerance", default_tol) or default_tol),
        })
    return feats


def _euclid(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def _best_match_score(mol_feats: List[Dict[str, Any]], query_feats: List[Dict[str, Any]]) -> Tuple[bool, float, List[Dict[str, Any]]]:
    # Greedy per-feature nearest neighbor of matching type within tolerance
    matched_details: List[Dict[str, Any]] = []
    d2s: List[float] = []
    for q in query_feats:
        q_type = _normalize_feature_type(q["feature_type"])  # normalized RDKit family name
        q_pt = (q["x"], q["y"], q["z"])
        tol = float(q.get("tolerance", 1.0))
        candidates = [m for m in mol_feats if m["type"].lower() == q_type]
        if not candidates:
            return False, float("inf"), []
        # pick nearest
        best = None
        best_d = float("inf")
        for m in candidates:
            m_pt = (m["x"], m["y"], m["z"])
            d = _euclid(q_pt, m_pt)
            if d < best_d:
                best_d = d
                best = m
        if best is None or best_d > tol:
            return False, float("inf"), []
        matched_details.append({
            "feature_type": q["feature_type"],
            "query_point": q_pt,
            "mol_point": (best["x"], best["y"], best["z"]),
            "distance": best_d,
        })
        d2s.append(best_d ** 2)
    rmsd = math.sqrt(sum(d2s) / len(d2s)) if d2s else 0.0
    return True, rmsd, matched_details


def _normalize_feature_type(t: str) -> str:
    key = str(t).strip().lower()
    mapping = {
        "hbd": "donor",
        "donor": "donor",
        "hba": "acceptor",
        "acceptor": "acceptor",
        "aromatic": "aromatic",
        "hydrophobic": "hydrophobe",
        "hydrophobe": "hydrophobe",
        "posionizable": "posionizable",
        "negionizable": "negionizable",
        "positive": "posionizable",
        "negative": "negionizable",
    }
    return mapping.get(key, key)


def _evaluate_molecule_3d(smiles: str, query_feats: List[Dict[str, Any]], n_conformers: int, seed: int = 0) -> Dict[str, Any]:
    _ensure_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"smiles": smiles, "match": False, "error": "invalid_smiles"}
    mol = Chem.AddHs(mol)
    try:
        conf_ids = _embed_and_optimize_conformers(mol, n_conformers=n_conformers, seed=seed)
    except Exception as e:
        return {"smiles": smiles, "match": False, "error": f"embed_failed: {e}"}
    best = {"match": False, "rmsd": float("inf"), "conformer_id": None, "details": []}
    for cid in conf_ids:
        try:
            mol.SetProp("_Name", "mol")
            _ = mol.GetConformer(cid)
        except Exception:
            continue
        feats = _mol_features_3d(mol, conf_id=int(cid))
        ok, rmsd, details = _best_match_score(feats, query_feats)
        if ok and rmsd < best["rmsd"]:
            best = {"match": True, "rmsd": rmsd, "conformer_id": int(cid), "details": details}
    if not best["match"]:
        return {"smiles": smiles, "match": False}
    return {"smiles": smiles, **best}


class PharmacophoreScreener:
    def __init__(self, mode: str = "2D") -> None:
        if mode not in {"2D", "3D"}:
            raise ValueError("mode must be '2D' or '3D'")
        self.mode = mode

    # 2D API
    def screen(
        self,
        *,
        query_smiles: Optional[str] = None,
        pharmacophore_csv: Optional[str] = None,
        library_csv: str,
        similarity_threshold: float = 0.0,
        radius: int = 2,
        n_bits: int = 2048,
        n_conformers: int = 10,
        n_workers: int = 1,
        batch_size: Optional[int] = None,
        chunksize: Optional[int] = None,
        seed: int = 0,
    ) -> ScreeningResult:
        """
        Run screening.

        - mode=2D requires query_smiles.
        - mode=3D requires pharmacophore_csv.
        Returns a ScreeningResult wrapping a DataFrame.
        """
        if self.mode == "2D":
            if not query_smiles:
                raise ValueError("query_smiles is required for 2D screening")
            return self._screen_2d(
                query_smiles=query_smiles,
                library_csv=library_csv,
                similarity_threshold=similarity_threshold,
                radius=radius,
                n_bits=n_bits,
                n_workers=n_workers,
                batch_size=batch_size,
                chunksize=chunksize,
            )
        else:
            if not pharmacophore_csv:
                raise ValueError("pharmacophore_csv is required for 3D screening")
            return self._screen_3d(
                pharmacophore_csv=pharmacophore_csv,
                library_csv=library_csv,
                n_conformers=n_conformers,
                n_workers=n_workers,
                batch_size=batch_size,
                chunksize=chunksize,
                seed=seed,
            )

    def _screen_2d(
        self,
        *,
        query_smiles: str,
        library_csv: str,
        similarity_threshold: float,
        radius: int,
        n_bits: int,
        n_workers: int,
        batch_size: Optional[int],
        chunksize: Optional[int],
    ) -> ScreeningResult:
        _ensure_rdkit()
        logger.info("Starting 2D screening")
        df_iter: Iterable[pd.DataFrame]
        if chunksize and chunksize > 0:
            df_iter = pd.read_csv(library_csv, chunksize=chunksize)
        else:
            df_iter = [pd.read_csv(library_csv)]

        frames: List[pd.DataFrame] = []
        for chunk in df_iter:
            if "smiles" not in chunk.columns:
                raise ValueError("library_csv must contain a 'smiles' column")
            n = len(chunk)
            if n == 0:
                continue
            bs = batch_size or _auto_batch_size(n, n_workers)
            use_ray = _init_ray_if_needed(n_workers)
            if use_ray:
                @ray.remote
                def _remote(start: int, end: int) -> pd.DataFrame:
                    sub = chunk.iloc[start:end].copy()
                    return _compute_similarity_batch(query_smiles, sub, radius, n_bits)
                pending = []
                for start, end in _chunk_indices(n, bs):
                    pending.append(_remote.remote(start, end))
                parts = ray.get(pending)
                part_df = pd.concat(parts, ignore_index=True)
            else:
                parts = []
                for start, end in _chunk_indices(n, bs):
                    sub = chunk.iloc[start:end].copy()
                    parts.append(_compute_similarity_batch(query_smiles, sub, radius, n_bits))
                part_df = pd.concat(parts, ignore_index=True)
            if similarity_threshold > 0:
                part_df = part_df[part_df["similarity"].fillna(0) >= similarity_threshold]
            frames.append(part_df)

        result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        # Ranking
        if "similarity" in result.columns:
            result = result.sort_values(by="similarity", ascending=False, na_position="last").reset_index(drop=True)
        logger.info("2D screening completed: %d rows", len(result))
        return ScreeningResult(result)

    # Optional visualization helpers
    def plot_2d(self, smiles: str, out_path: str, size: Tuple[int, int] = (300, 300)) -> None:
        _ensure_rdkit()
        from rdkit.Chem import Draw
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES for plotting")
        img = Draw.MolToImage(mol, size=size)
        img.save(out_path)

    def write_sdf(self, smiles: str, out_path: str, n_conformers: int = 1, seed: int = 0) -> None:
        _ensure_rdkit()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES for SDF export")
        mol = Chem.AddHs(mol)
        _embed_and_optimize_conformers(mol, n_conformers=n_conformers, seed=seed)
        writer = Chem.SDWriter(out_path)
        for cid in range(mol.GetNumConformers()):
            writer.write(mol, confId=cid)
        writer.close()

    def _screen_3d(
        self,
        *,
        pharmacophore_csv: str,
        library_csv: str,
        n_conformers: int,
        n_workers: int,
        batch_size: Optional[int],
        chunksize: Optional[int],
        seed: int,
    ) -> ScreeningResult:
        _ensure_rdkit()
        logger.info("Starting 3D screening")
        query_feats = _parse_pharmacophore_csv(pharmacophore_csv)
        df_iter: Iterable[pd.DataFrame]
        if chunksize and chunksize > 0:
            df_iter = pd.read_csv(library_csv, chunksize=chunksize)
        else:
            df_iter = [pd.read_csv(library_csv)]

        frames: List[pd.DataFrame] = []
        for chunk in df_iter:
            if "smiles" not in chunk.columns:
                raise ValueError("library_csv must contain a 'smiles' column")
            n = len(chunk)
            if n == 0:
                continue
            bs = batch_size or _auto_batch_size(n, n_workers)
            use_ray = _init_ray_if_needed(n_workers)
            if use_ray:
                @ray.remote
                def _remote_eval(sm: str) -> Dict[str, Any]:
                    return _evaluate_molecule_3d(sm, query_feats, n_conformers, seed)
                futures = [_remote_eval.remote(sm) for sm in chunk["smiles"].astype(str).tolist()]
                results = ray.get(futures)
            else:
                results = [_evaluate_molecule_3d(sm, query_feats, n_conformers, seed) for sm in chunk["smiles"].astype(str).tolist()]
            df_res = pd.DataFrame(results)
            frames.append(pd.concat([chunk.reset_index(drop=True), df_res], axis=1))

        result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        # Keep useful columns forward
        cols = list(result.columns)
        ordered = [c for c in ["match", "rmsd", "conformer_id", "details", "error"] if c in cols]
        rest = [c for c in cols if c not in ordered]
        result = result[rest + ordered]
        logger.info("3D screening completed: %d rows", len(result))
        return ScreeningResult(result)
