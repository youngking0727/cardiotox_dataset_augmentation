"""
与 scripts/chembl_excel_full_enrichment.py 中相似度查询一致的策略：
多组分 SMILES（盐、溶剂等）在调用 ChEMBL /similarity 前改为单片段（按重原子数优先），
避免整条混合物字符串导致相似检索不稳定或无效。

依赖 RDKit；不可用时回退为原字符串。
"""

from __future__ import annotations

from typing import Optional

try:
    from rdkit import Chem
except ImportError:
    Chem = None  # type: ignore[misc, assignment]


def inchikey_from_smiles(s: str) -> Optional[str]:
    """
    与 chembl_excel_full_enrichment.inchikey_from_smiles 一致。
    ChEMBL 支持 GET /molecule/{inchikey}.json 直接取主记录。
    """
    if Chem is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    try:
        from rdkit.Chem import inchi

        return inchi.MolToInchiKey(mol)
    except Exception:
        return None


def canonicalize_smiles(s: str) -> Optional[str]:
    if Chem is None:
        return None
    if not s or not str(s).strip():
        return None
    s = str(s).strip()
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def split_smiles_frags(s: str) -> list[str]:
    s = str(s).strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(".") if p.strip()]
    return parts if parts else [s]


def frag_heavy_atoms(smiles: str) -> int:
    if Chem is None:
        return 0
    m = Chem.MolFromSmiles(smiles)
    return m.GetNumHeavyAtoms() if m else 0


def ordered_frags_for_lookup(canonical_full: str) -> list[str]:
    """多组分时按重原子数降序取可 canonicalize 的片段，去重。"""
    frags = split_smiles_frags(canonical_full)
    if len(frags) <= 1:
        return frags
    ranked = sorted(frags, key=lambda f: -frag_heavy_atoms(f))
    seen: set[str] = set()
    out: list[str] = []
    for f in ranked:
        c = canonicalize_smiles(f)
        if not c or c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def smiles_for_similarity_query(canonical_full: str) -> str:
    """
    与 chembl_excel_full_enrichment.smiles_for_similarity 一致：
    相似度 API 使用单组分 SMILES；多组分时取重原子数最高的片段（canonical 后）。
    """
    frags = ordered_frags_for_lookup(canonical_full)
    if not frags:
        return canonical_full
    if len(frags) == 1:
        return frags[0]
    return frags[0]
