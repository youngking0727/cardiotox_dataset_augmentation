"""M2: PhysChemCalculator — ChEMBL 基线 + RDKit 描述符 + 规则工程特征（与 config/Features.json 对齐）。"""

from __future__ import annotations

import json
import logging
import math
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from schemas import PhysChemProperties, SourceInfo
from utils.chembl_client import ChEMBLClient
from utils.cache import get_global_cache

logger = logging.getLogger(__name__)

# 暂时关闭：pkasolver 补 pKa，以及由 logP+pKa 估算 logD(7.4)。改为 True 后恢复。
_ENRICH_PKA_LOGD = False

_FEATURES_JSON = Path(__file__).resolve().parent.parent / "config" / "Features.json"


def _load_feature_keys() -> List[str]:
    with open(_FEATURES_JSON, encoding="utf-8") as f:
        data = json.load(f)
    return list(data.keys())


REQUIRED_FEATURE_KEYS: List[str] = _load_feature_keys()

# 与 Features.json 对齐：RDKit 直接计算 / 规则与 SMARTS
RDKIT_DESCRIPTOR_KEYS = {
    "NumHDonors",
    "NumHAcceptors",
    "FractionCSP3",
    "HeavyAtomCount",
    "MolMR",
    "BertzCT",
    "Chi0v",
    "Chi1v",
    "HallKierAlpha",
    "MolWt",
    "logP",
    "cLogP",
    "topological_polar_surface_area",
    "num_rotatable",
    "formal_charge",
    "n_heteroatoms",
    "n_aromatic_atoms",
    "num_aromatic_rings",
    # 新增：需求文档不重复的 5 项（与 Features.json 对齐）
    "logd_7_4",
    "basic_pka",
    "acidic_pka",
    "ro5_violations",
    "qed",
}

# 工程特征：通过 SMARTS 匹配、规则计算得到的特征
# 隐式推导：ENGINEERED_FEATURE_KEYS = REQUIRED_FEATURE_KEYS - RDKIT_DESCRIPTOR_KEYS
# 这意味着每当 Features.json 更新时，代码会自动同步，但可能掩盖遗漏问题
ENGINEERED_FEATURE_KEYS = [k for k in REQUIRED_FEATURE_KEYS if k not in RDKIT_DESCRIPTOR_KEYS]

# RDKit 描述符：连续值一律 float；计数字段一律 int
RDKIT_FLOAT_KEYS = {
    "MolWt",
    "logP",
    "cLogP",
    "topological_polar_surface_area",
    "MolMR",
    "BertzCT",
    "Chi0v",
    "Chi1v",
    "HallKierAlpha",
    "FractionCSP3",
}

# 工程特征：显式计数字段 int（其余 has_* / pharmacophore_match_hERG_like 为 0/1）
ENGINEERED_INT_COUNT_KEYS = {
    "num_OH",
    "n_basic_N",
    "n_quat_N",
    "n_tertiary_N",
    "n_benzene_like_rings",
    "min_basicN_to_aromatic_ring_dist",
}


def _rule_hit_01(v: Any) -> int:
    """规则命中：统一 0/1。"""
    if v is True or v == 1:
        return 1
    if v is False or v == 0:
        return 0
    if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
        return 1 if int(v) else 0
    s = str(v).strip().lower()
    if s in ("1", "true", "yes"):
        return 1
    return 0


def _normalize_rdkit_descriptor_types(rd: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in rd.items():
        if k in RDKIT_FLOAT_KEYS:
            if isinstance(v, float) and math.isnan(v):
                out[k] = None  # nan 不是合法 JSON，转换为 None
            else:
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    out[k] = 0.0
        else:
            if isinstance(v, float) and math.isnan(v):
                out[k] = None
            else:
                try:
                    out[k] = int(round(float(v)))
                except (TypeError, ValueError):
                    out[k] = 0
    return out


def _normalize_engineered_types(eng: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in eng.items():
        if k == "pharmacophore_details":
            out[k] = v
        elif k.startswith("has_") or k == "pharmacophore_match_hERG_like":
            out[k] = _rule_hit_01(v)
        elif k in ENGINEERED_INT_COUNT_KEYS:
            try:
                out[k] = int(round(float(v))) if v is not None else 0
            except (TypeError, ValueError):
                out[k] = 0
        else:
            out[k] = v
    return out


def _is_missing_value(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return False


def _chembl_si_float(v: Any, default: float = 0.0) -> SourceInfo:
    """ChEMBL 常返回 null；SourceInfo.value 不可为 None。"""
    if _is_missing_value(v):
        return SourceInfo(value=default, source="chembl")
    return SourceInfo(value=v, source="chembl")


def _chembl_si_int(v: Any, default: int = 0) -> SourceInfo:
    if _is_missing_value(v):
        return SourceInfo(value=default, source="chembl")
    return SourceInfo(value=v, source="chembl")


def _float_from_any(v: Any) -> Optional[float]:
    if _is_missing_value(v):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def estimate_logd_7_4(
    logp: Optional[float],
    acidic_pka: Optional[float] = None,
    basic_pka: Optional[float] = None,
    ph: float = 7.4,
) -> Optional[float]:
    """
    用 Henderson–Hasselbalch 近似估算 logD at 指定 pH（默认 7.4）。

    - 只有碱性中心：弱碱公式
    - 只有酸性中心：弱酸公式
    - 同时有酸/碱中心：简单两性近似
    - 都没有：logD = logP
    """
    if logp is None:
        return None

    try:
        p = 10 ** float(logp)

        if acidic_pka is not None and basic_pka is not None:
            denom = 1.0 + (10 ** (ph - acidic_pka)) + (10 ** (basic_pka - ph))
            d = p / denom
            return round(math.log10(d), 4)

        if acidic_pka is not None:
            denom = 1.0 + (10 ** (ph - acidic_pka))
            d = p / denom
            return round(math.log10(d), 4)

        if basic_pka is not None:
            denom = 1.0 + (10 ** (basic_pka - ph))
            d = p / denom
            return round(math.log10(d), 4)

        return round(float(logp), 4)
    except Exception:
        return None


def _int_from_any(v: Any) -> Optional[int]:
    x = _float_from_any(v)
    if x is None:
        return None
    return int(round(x))


def normalize_physchem_numeric_core(p: PhysChemProperties) -> PhysChemProperties:
    """
    将核心 SourceInfo 中 ChEMBL 常返回的字符串数值统一为 float/int，
    便于下游 pipeline 与 JSON 类型稳定（mw/logp/tpsa/qed 等为数字）。
    """

    def f(si: SourceInfo) -> SourceInfo:
        v = _float_from_any(si.value)
        if v is None:
            return SourceInfo(value=0.0, source=si.source)
        return SourceInfo(value=v, source=si.source)

    def i(si: SourceInfo) -> SourceInfo:
        iv = _int_from_any(si.value)
        if iv is None:
            return SourceInfo(value=0, source=si.source)
        return SourceInfo(value=iv, source=si.source)

    def opt_f(si: Optional[SourceInfo]) -> Optional[SourceInfo]:
        if si is None:
            return None
        v = _float_from_any(si.value)
        if v is None:
            return None
        return SourceInfo(value=v, source=si.source)

    return PhysChemProperties(
        mw=f(p.mw),
        logp=f(p.logp),
        logd_7_4=opt_f(p.logd_7_4),
        tpsa=f(p.tpsa),
        hbd=i(p.hbd),
        hba=i(p.hba),
        rotatable_bonds=i(p.rotatable_bonds),
        aromatic_rings=i(p.aromatic_rings),
        heavy_atoms=i(p.heavy_atoms),
        basic_pka=opt_f(p.basic_pka),
        acidic_pka=opt_f(p.acidic_pka),
        ro5_violations=i(p.ro5_violations),
        qed=f(p.qed),
        rdkit_descriptors=dict(getattr(p, "rdkit_descriptors", None) or {}),
        engineered_features=dict(getattr(p, "engineered_features", None) or {}),
    )


class PhysChemCalculator:
    """理化性质：ChEMBL 核心基线 + RDKit 补全 + Features.json 全量描述符/工程特征。"""

    def __init__(self, chembl_client: Optional[ChEMBLClient] = None):
        self.chembl_client = chembl_client or ChEMBLClient()
        self.cache = get_global_cache()
        self._chembl_molecule_cache: Dict[str, Dict[str, Any]] = {}  # 缓存 molecule 数据，避免重复 API 调用

    # ------------------------------------------------------------------ #
    # ChEMBL
    # ------------------------------------------------------------------ #
    def calculate_from_chembl(self, chembl_id: str) -> Optional[PhysChemProperties]:
        cache_key = f"physchem_chembl_{chembl_id}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return normalize_physchem_numeric_core(cached)

        logger.info("从ChEMBL获取理化性质: %s", chembl_id)
        molecule, _mol_ok = self.chembl_client.get_molecule_by_chembl_id(chembl_id)
        if not molecule or not isinstance(molecule, dict):
            logger.warning("未找到分子: %s", chembl_id)
            return None

        # 缓存 molecule 数据，避免后续 _resolve_working_smiles 重复调用 API
        self._chembl_molecule_cache[chembl_id] = molecule

        props = molecule.get("molecule_properties")
        if not props:
            logger.warning("分子无可用性质: %s", chembl_id)
            return None

        try:
            mw_raw = props.get("full_mwt") or props.get("molweight")
            ro5_raw = props.get("num_ro5_violations")
            if ro5_raw is None:
                ro5_raw = 0
            result = PhysChemProperties(
                mw=_chembl_si_float(mw_raw, 0.0),
                logp=_chembl_si_float(props.get("alogp"), 0.0),
                logd_7_4=None,
                tpsa=_chembl_si_float(props.get("psa"), 0.0),
                hbd=_chembl_si_int(props.get("hbd"), 0),
                hba=_chembl_si_int(props.get("hba"), 0),
                rotatable_bonds=_chembl_si_int(props.get("rtb"), 0),
                aromatic_rings=_chembl_si_int(props.get("aromatic_rings"), 0),
                heavy_atoms=_chembl_si_int(props.get("heavy_atoms"), 0),
                basic_pka=None,
                acidic_pka=None,
                ro5_violations=_chembl_si_int(ro5_raw, 0),
                qed=_chembl_si_float(props.get("qed_weighted"), 0.0),
                rdkit_descriptors={},
                engineered_features={},
            )
            can_smi = (molecule.get("molecule_structures") or {}).get("canonical_smiles")
            if can_smi:
                self.cache.set(f"physchem_chembl_smiles_{chembl_id}", can_smi)
            result = normalize_physchem_numeric_core(result)
            self.cache.set(cache_key, result)
            return result
        except Exception as e:
            logger.warning("解析ChEMBL理化性质失败: %s, 错误: %s", chembl_id, e)
            return None

    # ------------------------------------------------------------------ #
    # RDKit bundle（公开 API）
    # ------------------------------------------------------------------ #
    def calculate_rdkit_feature_bundle(self, smiles: str) -> Dict[str, Any]:
        """
        返回与 config/Features.json 键集合对齐的完整特征：
        - rdkit_descriptors: RDKit 计算描述符
        - engineered_features: 规则 / SMARTS / hERG-like 等
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import (
                Crippen,
                Descriptors,
                GraphDescriptors,
                Lipinski,
                QED,
                rdMolDescriptors,
            )
        except ImportError as e:
            raise RuntimeError("需要安装 RDKit") from e

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"无效 SMILES: {smiles}")

        # --- RDKit 数值描述符 ---
        logp_val = float(Crippen.MolLogP(mol))
        tpsa_val = float(Descriptors.TPSA(mol))
        molwt = float(Descriptors.MolWt(mol))
        n_heavy = int(rdMolDescriptors.CalcNumHeavyAtoms(mol))
        n_rot = int(Lipinski.NumRotatableBonds(mol))
        n_h_don = int(Lipinski.NumHDonors(mol))
        n_h_acc = int(Lipinski.NumHAcceptors(mol))
        fsp3 = float(Lipinski.FractionCSP3(mol))
        mol_mr = float(Crippen.MolMR(mol))
        hall_kier = float(Descriptors.HallKierAlpha(mol))
        fcharge = int(Chem.GetFormalCharge(mol))
        n_het = int(rdMolDescriptors.CalcNumHeteroatoms(mol))

        try:
            bertz = float(GraphDescriptors.BertzCT(mol))
        except Exception:
            bertz = float("nan")
        try:
            chi0 = float(GraphDescriptors.Chi0v(mol))
            chi1 = float(GraphDescriptors.Chi1v(mol))
        except Exception:
            chi0 = float("nan")
            chi1 = float("nan")

        n_arom_rings = int(Lipinski.NumAromaticRings(mol))
        n_arom_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())

        rdkit_descriptors: Dict[str, Any] = {
            "NumHDonors": n_h_don,
            "NumHAcceptors": n_h_acc,
            "FractionCSP3": fsp3,
            "HeavyAtomCount": n_heavy,
            "MolMR": mol_mr,
            "BertzCT": bertz,
            "Chi0v": chi0,
            "Chi1v": chi1,
            "HallKierAlpha": hall_kier,
            "MolWt": molwt,
            "logP": logp_val,
            "cLogP": logp_val,  # 与 logP 同值，保持与 Features.json 兼容
            "topological_polar_surface_area": tpsa_val,
            "num_rotatable": n_rot,
            "formal_charge": fcharge,
            "n_heteroatoms": n_het,
            "n_aromatic_atoms": n_arom_atoms,
            "num_aromatic_rings": n_arom_rings,
            # 新增：需求文档不重复的 5 项（与 Features.json 54 项对齐）
            "ro5_violations": self._count_ro5_violations(mol),
            "qed": float(QED.qed(mol)),
            "logd_7_4": None,  # 需要 pkasolver，默认关闭
            "basic_pka": None,  # 需要 pkasolver，默认关闭
            "acidic_pka": None,  # 需要 pkasolver，默认关闭
        }

        rdkit_descriptors = _normalize_rdkit_descriptor_types(rdkit_descriptors)

        engineered = self._compute_engineered_features(mol, rdkit_descriptors)

        union = set(rdkit_descriptors) | set(engineered)
        missing = set(REQUIRED_FEATURE_KEYS) - union
        if missing:
            raise RuntimeError(
                f"Features.json 键未全部覆盖: missing={sorted(missing)}"
            )

        return {"rdkit_descriptors": rdkit_descriptors, "engineered_features": engineered}

    def _compute_engineered_features(
        self, mol: Any, rdkit_descriptors: Dict[str, Any]
    ) -> Dict[str, Any]:
        from rdkit import Chem

        # SMARTS 预编译（惰性）
        def _pat(s: str):
            p = Chem.MolFromSmarts(s)
            return p

        # 负离子/可电离酸
        neg_ion = _pat("[O-][C,S,N]=O") or _pat("[S-]")
        has_neg = bool(neg_ion and mol.HasSubstructMatch(neg_ion))

        n_h_acc = int(rdkit_descriptors["NumHAcceptors"])
        tpsa = float(rdkit_descriptors["topological_polar_surface_area"])
        n_rot = int(rdkit_descriptors["num_rotatable"])

        has_hba = n_h_acc > 0
        has_polar = tpsa > 20.0
        has_flex = n_rot > 4

        # 脂肪族环氮（碱性氮在环内）
        basic_n_ring = _pat("[#7;R;!a]")
        has_basic_n_ring = bool(basic_n_ring and mol.HasSubstructMatch(basic_n_ring))

        # OH 计数（脂肪醇/酚）
        oh_pat = _pat("[OX2H]")
        num_oh = (
            len(mol.GetSubstructMatches(oh_pat)) if oh_pat else 0
        )

        # 含氮计数与类型（简化）
        n_basic_n, n_tert_n, n_quat_n = self._count_nitrogen_classes(mol)

        # 环系 SMARTS
        pip = _pat("C1CNCCN1")
        pipz = _pat("N1CCNCC1")
        pyr = _pat("c1ccncc1")
        quin = _pat("c1ccc2ncccc2c1")
        isoquin = _pat("c1ccc2cnccc2c1")
        indole = _pat("c1ccc2[nH]ccc2c1")

        has_pip = int(bool(pip and mol.HasSubstructMatch(pip)))
        has_pipz = int(bool(pipz and mol.HasSubstructMatch(pipz)))
        has_pyr = int(bool(pyr and mol.HasSubstructMatch(pyr)))
        has_quin = int(bool(quin and mol.HasSubstructMatch(quin)))
        has_isoq = int(bool(isoquin and mol.HasSubstructMatch(isoquin)))
        has_ind = int(bool(indole and mol.HasSubstructMatch(indole)))

        # “_like” 与 has_* 同值（与示例 JSON 命名并存）
        n_benzene_like = self._count_benzene_like_rings(mol)
        has_bis_aryl = int(self._has_biaryl(mol))
        pos_fc = int(Chem.GetFormalCharge(mol) > 0)

        has_tert_amine = int(self._has_tertiary_amine(mol))
        has_quat_am = int(self._has_quaternary_ammonium(mol))

        herg_match, herg_detail = self._herg_like_pharmacophore(mol)
        min_dist = self._min_basic_n_to_aromatic_dist(mol)

        has_bis_aryl_basic = int(bool(has_bis_aryl and n_basic_n >= 1))

        h_grp = self._hydrophobic_group_count(mol)
        has_three_or_four_hg = int(3 <= h_grp <= 4)

        eng: Dict[str, Any] = {
            "has_neg_ion_group": int(bool(has_neg)),
            "has_hbond_acceptor": int(has_hba),
            "has_polar_group": int(has_polar),
            "has_flexible_chain": int(has_flex),
            "has_basicN_any_ring": int(has_basic_n_ring),
            "num_OH": int(num_oh),
            "has_piperazine": has_pipz,
            "has_piperidine": has_pip,
            "has_quaternary_N": int(n_quat_n > 0),
            "has_tertiary_N": int(n_tert_n > 0),
            "has_quinoline": has_quin,
            "has_isoquinoline": has_isoq,
            "has_indole": has_ind,
            "has_pyridine": has_pyr,
            "n_benzene_like_rings": n_benzene_like,
            "has_bis_aryl": has_bis_aryl,
            "has_positive_formal_charge": pos_fc,
            "n_basic_N": n_basic_n,
            "n_tertiary_N": n_tert_n,
            "n_quat_N": n_quat_n,
            "min_basicN_to_aromatic_ring_dist": min_dist,
            "pharmacophore_match_hERG_like": herg_match,
            "pharmacophore_details": herg_detail,
            "has_piperazine_like": has_pipz,
            "has_piperidine_like": has_pip,
            "has_tertiary_amine": has_tert_amine,
            "has_quaternary_ammonium": has_quat_am,
            "has_quinoline_like": has_quin,
            "has_indole_like": has_ind,
            "has_bis_aryl_and_basicN": has_bis_aryl_basic,
            "has_three_or_four_hydrophobic_groups": has_three_or_four_hg,
        }

        for k in ENGINEERED_FEATURE_KEYS:
            if k not in eng:
                if k == "pharmacophore_details":
                    eng[k] = {"reason": "none", "basic_ns": [], "n_rings": 0}
                elif k == "num_OH":
                    eng[k] = 0
                else:
                    eng[k] = 0
        ordered = {k: eng[k] for k in ENGINEERED_FEATURE_KEYS}
        return _normalize_engineered_types(ordered)

    def _count_nitrogen_classes(self, mol) -> Tuple[int, int, int]:
        from rdkit import Chem

        n_basic = 0
        n_tert = 0
        n_quat = 0
        for a in mol.GetAtoms():
            if a.GetAtomicNum() != 7:
                continue
            chg = a.GetFormalCharge()
            if chg == 1 and a.GetTotalDegree() == 4:
                n_quat += 1
                continue
            if self._is_basic_aliphatic_n(a, mol):
                n_basic += 1
                if a.GetTotalDegree() == 3 and a.GetTotalNumHs() == 0:
                    n_tert += 1
        return n_basic, n_tert, n_quat

    def _is_basic_aliphatic_n(self, atom, mol) -> bool:
        """
        判断脂肪氮是否为碱性氮（可接受质子）。
        排除：芳香氮、酰胺氮、硝基氮等。
        """
        from rdkit import Chem

        if atom.GetAtomicNum() != 7:
            return False
        # 芳香氮排除
        if atom.GetIsAromatic():
            return False

        # 酰胺氮排除：氮原子直接连接羰基碳 [N]-[C]=[O]
        # SMARTS: N-C(=O) 或 N-C(=S)
        amide_pat = Chem.MolFromSmarts("[NX3][CX3]=[OX1]")
        if amide_pat and mol.HasSubstructMatch(amide_pat):
            # 检查当前氮是否在酰胺模式中
            for match in mol.GetSubstructMatches(amide_pat):
                if atom.GetIdx() == match[0]:  # 第一个原子是氮
                    return False

        # 排除连接到大原子（O,S）的sp2碳的氮（可能是酰胺、酯等）
        for nb in atom.GetNeighbors():
            if nb.GetAtomicNum() == 8 or nb.GetAtomicNum() == 16:
                # 检查氮-碳-氧的连接模式
                if nb.GetHybridization() == Chem.HybridizationType.SP2:
                    if nb.GetTotalDegree() == 2:
                        return False
        return True

    def _count_benzene_like_rings(self, mol) -> int:
        from rdkit import Chem

        ri = mol.GetRingInfo()
        n = 0
        for ring in ri.AtomRings():
            if len(ring) == 6 and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                n += 1
        return n

    def _has_biaryl(self, mol) -> bool:
        """
        检测联苯结构（两个芳香环通过单键相连）。
        使用SMARTS而非SMILES，避免表示差异导致漏检。
        SMARTS: 两个芳香碳环通过单键连接
        """
        from rdkit import Chem

        # SMARTS: 任意两个芳香环通过单键相连 (c:c 表示芳香碳之间的单键)
        pat = Chem.MolFromSmarts("c-c")
        if pat is None:
            return False
        # 检查分子中是否存在 c-c（两个相连的芳香碳）
        return mol.HasSubstructMatch(pat)

    def _has_tertiary_amine(self, mol) -> bool:
        from rdkit import Chem

        pat = Chem.MolFromSmarts("[NX3;H0;!$(N=O);!$(NC=O)]")
        return bool(pat and mol.HasSubstructMatch(pat))

    def _has_quaternary_ammonium(self, mol) -> bool:
        from rdkit import Chem

        pat = Chem.MolFromSmarts("[N+;X4]")
        return bool(pat and mol.HasSubstructMatch(pat))

    def _herg_like_pharmacophore(self, mol) -> Tuple[int, Dict[str, Any]]:
        from rdkit.Chem import Lipinski

        n_arom_rings = int(Lipinski.NumAromaticRings(mol))
        basic_idxs: List[int] = []
        for a in mol.GetAtoms():
            if a.GetAtomicNum() == 7 and self._is_basic_aliphatic_n(a, mol):
                basic_idxs.append(a.GetIdx())
        n_basic = len(basic_idxs)
        match = 1 if (n_arom_rings >= 2 and n_basic >= 1) else 0
        if match:
            reason = "two_aromatic_rings_and_basic_N"
        elif n_arom_rings < 2:
            reason = "no_two_aromatic_rings_or_no_basic_N"
        else:
            reason = "no_basic_N"
        detail = {
            "reason": reason,
            "basic_ns": basic_idxs,
            "n_rings": n_arom_rings,
        }
        return match, detail

    def _min_basic_n_to_aromatic_dist(self, mol) -> int:
        from rdkit import Chem

        basic_idx = [
            a.GetIdx()
            for a in mol.GetAtoms()
            if a.GetAtomicNum() == 7 and self._is_basic_aliphatic_n(a, mol)
        ]
        arom_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]
        if not basic_idx or not arom_idx:
            return -1

        adj: Dict[int, List[int]] = {i: [] for i in range(mol.GetNumAtoms())}
        for b in mol.GetBonds():
            u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            adj[u].append(v)
            adj[v].append(u)

        best = 10**9
        for s in basic_idx:
            q = deque([(s, 0)])
            seen = {s}
            while q:
                u, d = q.popleft()
                if u in arom_idx:
                    best = min(best, d)
                if d > 20:
                    continue
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v)
                        q.append((v, d + 1))
        return int(best if best < 10**9 else -1)

    def _hydrophobic_group_count(self, mol) -> int:
        """疏水片段粗计数：芳香环数 + 端甲基数（[CX4;H3]），供 has_three_or_four_hydrophobic_groups。"""
        from rdkit import Chem
        from rdkit.Chem import Lipinski

        p = Chem.MolFromSmarts("[CX4;H3]")
        n_me = len(mol.GetSubstructMatches(p)) if p else 0
        n_ar = int(Lipinski.NumAromaticRings(mol))
        return n_me + n_ar

    # ------------------------------------------------------------------ #
    # RDKit-only 核心（PhysChemProperties）
    # ------------------------------------------------------------------ #
    def calculate_from_smiles(self, smiles: str) -> Optional[PhysChemProperties]:
        try:
            bundle = self.calculate_rdkit_feature_bundle(smiles)
        except Exception as e:
            logger.warning("RDKit bundle 失败: %s", e)
            return None
        rd = bundle["rdkit_descriptors"]
        eng = bundle["engineered_features"]
        try:
            from rdkit import Chem
            from rdkit.Chem import QED

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
        except ImportError:
            logger.error("RDKit 未安装")
            return None

        ro5 = self._count_ro5_violations(mol)
        return normalize_physchem_numeric_core(
            PhysChemProperties(
                mw=SourceInfo(value=rd["MolWt"], source="rdkit"),
                logp=SourceInfo(value=rd["logP"], source="rdkit"),
                logd_7_4=None,
                tpsa=SourceInfo(
                    value=rd["topological_polar_surface_area"], source="rdkit"
                ),
                hbd=SourceInfo(value=rd["NumHDonors"], source="rdkit"),
                hba=SourceInfo(value=rd["NumHAcceptors"], source="rdkit"),
                rotatable_bonds=SourceInfo(value=rd["num_rotatable"], source="rdkit"),
                aromatic_rings=SourceInfo(
                    value=int(rd["num_aromatic_rings"]), source="rdkit"
                ),
                heavy_atoms=SourceInfo(value=rd["HeavyAtomCount"], source="rdkit"),
                basic_pka=None,
                acidic_pka=None,
                ro5_violations=SourceInfo(value=ro5, source="rdkit"),
                qed=SourceInfo(value=float(QED.qed(mol)), source="rdkit"),
                rdkit_descriptors=rd,
                engineered_features=eng,
            )
        )

    def _count_ro5_violations(self, mol) -> int:
        try:
            from rdkit.Chem import Crippen, Descriptors, Lipinski

            v = 0
            if Descriptors.MolWt(mol) > 500:
                v += 1
            if Crippen.MolLogP(mol) > 5:
                v += 1
            if Lipinski.NumHDonors(mol) > 5:
                v += 1
            if Lipinski.NumHAcceptors(mol) > 10:
                v += 1
            return v
        except Exception as e:
            logger.warning("RO5 计算失败: %s", e)
            return 0

    def calculate_pka(self, smiles: str) -> Optional[Dict[str, float]]:
        """
        pKa 为可选依赖：优先旧版 API ``PkaCalculator.calc_pka``；
        否则尝试 mayrf/pkasolver 自带的 ``query.calculate_microstate_pka_values``（需 torch 等）。
        """
        # --- 旧版 / 兼容 fork：PkaCalculator ---
        try:
            from pkasolver import PkaCalculator  # type: ignore[attr-defined]
        except ImportError:
            PkaCalculator = None  # type: ignore[misc, assignment]

        if PkaCalculator is not None:
            try:
                calculator = PkaCalculator()
                pka_results = calculator.calc_pka(smiles)
                if not pka_results:
                    return None
                sorted_pkas = sorted(pka_results, key=lambda x: x.get("pka", 0))
                basic_pka = None
                acidic_pka = None
                for pka_info in sorted_pkas:
                    pka_type = pka_info.get("type", "")
                    pka_val = pka_info.get("pka")
                    if pka_type == "basic" and basic_pka is None:
                        basic_pka = pka_val
                    elif pka_type == "acidic" and acidic_pka is None:
                        acidic_pka = pka_val
                return {"basic_pka": basic_pka, "acidic_pka": acidic_pka}
            except Exception as e:
                logger.debug("PkaCalculator pKa 失败，尝试 query 后端: %s", e)

        # --- mayrf/pkasolver 主分支：Graph 模型 ---
        try:
            from rdkit import Chem
            from pkasolver.query import calculate_microstate_pka_values
        except ImportError as e:
            logger.warning(
                "pKa 不可用：无法 import pkasolver.query（缺 svgutils/cairosvg 或 torch 等）: %s",
                e,
            )
            return None

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            states = calculate_microstate_pka_values(mol)
            if not states:
                logger.warning(
                    "pkasolver 未返回任何 microstate pKa，smiles=%s",
                    smiles,
                )
                return None
            pkas = [float(s.pka) for s in states]
            low = [p for p in pkas if p < 7.0]
            high = [p for p in pkas if p >= 7.0]
            return {
                "acidic_pka": min(low) if low else None,
                "basic_pka": max(high) if high else None,
            }
        except Exception as e:
            # 常见：bundled .pt 权重与当前 torch_geometric 版 GIN 结构不一致（Missing/unexpected keys in state_dict）
            logger.warning(
                "pkasolver.query pKa 计算失败（若见 state_dict 不匹配，需对齐 pkasolver 与 torch-geometric 版本或重装官方权重）: %s",
                e,
            )
            return None

    def calculate_pka_from_smiles(self, smiles: str) -> Dict[str, Optional[float]]:
        """
        从 SMILES 预测最强酸性 / 碱性 pKa。

        若上游 ``QueryModel`` 提供 ``predict(smiles)`` 且返回带 ``pKa``/``type`` 的条目则优先使用；
        否则回退到 :meth:`calculate_pka`（PkaCalculator / microstate）。
        """
        empty: Dict[str, Optional[float]] = {"basic_pka": None, "acidic_pka": None}
        if not (smiles or "").strip():
            return empty

        try:
            from pkasolver.query import QueryModel  # type: ignore[import-untyped]

            model = QueryModel()
            predict_fn = getattr(model, "predict", None)
            if callable(predict_fn):
                result = predict_fn(smiles)
                if result is not None:
                    acidic_values: List[float] = []
                    basic_values: List[float] = []
                    items = result if isinstance(result, (list, tuple)) else (result,)
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        pka_value = item.get("pKa")
                        if pka_value is None:
                            pka_value = item.get("pka")
                        pka_type = str(item.get("type", "")).lower()
                        if pka_value is None:
                            continue
                        try:
                            pka_value = float(pka_value)
                        except (TypeError, ValueError):
                            continue
                        if "acid" in pka_type:
                            acidic_values.append(pka_value)
                        elif "base" in pka_type:
                            basic_values.append(pka_value)
                    if acidic_values or basic_values:
                        return {
                            "basic_pka": max(basic_values) if basic_values else None,
                            "acidic_pka": min(acidic_values) if acidic_values else None,
                        }
        except Exception as e:
            logger.debug("QueryModel.predict pKa 不可用: %s", e)

        legacy = self.calculate_pka(smiles)
        if not legacy:
            logger.warning("pkasolver pKa 预测失败或未返回结果, smiles=%s", smiles)
            return empty
        return {
            "basic_pka": legacy.get("basic_pka"),
            "acidic_pka": legacy.get("acidic_pka"),
        }

    def _enrich_pka_logd(
        self, props: PhysChemProperties, smiles: str
    ) -> PhysChemProperties:
        """
        富集 pKa 和 logD 信息。返回新的 PhysChemProperties 对象（immutable）。
        """
        if not _ENRICH_PKA_LOGD:
            return props

        # 创建新对象，避免 mutation
        new_props = PhysChemProperties(
            mw=props.mw,
            logp=props.logp,
            logd_7_4=props.logd_7_4,
            tpsa=props.tpsa,
            hbd=props.hbd,
            hba=props.hba,
            rotatable_bonds=props.rotatable_bonds,
            aromatic_rings=props.aromatic_rings,
            heavy_atoms=props.heavy_atoms,
            basic_pka=props.basic_pka,
            acidic_pka=props.acidic_pka,
            ro5_violations=props.ro5_violations,
            qed=props.qed,
            rdkit_descriptors=props.rdkit_descriptors,
            engineered_features=props.engineered_features,
        )

        pka_part = self.calculate_pka_from_smiles(smiles)
        basic_pka_value = pka_part.get("basic_pka")
        acidic_pka_value = pka_part.get("acidic_pka")

        logp_value: Optional[float] = None
        if props.logp is not None and getattr(props.logp, "value", None) is not None:
            logp_value = _float_from_any(props.logp.value)

        logd_value = estimate_logd_7_4(
            logp=logp_value,
            acidic_pka=acidic_pka_value,
            basic_pka=basic_pka_value,
            ph=7.4,
        )

        if basic_pka_value is not None:
            new_props.basic_pka = SourceInfo(value=float(basic_pka_value), source="pkasolver")
        if acidic_pka_value is not None:
            new_props.acidic_pka = SourceInfo(
                value=float(acidic_pka_value), source="pkasolver"
            )
        if logd_value is not None:
            new_props.logd_7_4 = SourceInfo(
                value=logd_value, source="estimated_from_logp_and_pka"
            )
        return new_props

    def _resolve_working_smiles(
        self,
        chembl_id: Optional[str],
        smiles: Optional[str],
        chembl_part: Optional[PhysChemProperties],
    ) -> Optional[str]:
        # 优先从实例变量缓存中获取 molecule（calculate_from_chembl 已缓存）
        if chembl_id:
            cached_mol = self._chembl_molecule_cache.get(chembl_id)
            if cached_mol and isinstance(cached_mol, dict):
                s = (cached_mol.get("molecule_structures") or {}).get("canonical_smiles")
                if s:
                    return str(s)

            # 尝试从缓存（文件系统）中获取
            s = self.cache.get(f"physchem_chembl_smiles_{chembl_id}")
            if s:
                return str(s)

            # 最后才调用 API
            mol, _ = self.chembl_client.get_molecule_by_chembl_id(chembl_id)
            if isinstance(mol, dict):
                s = (mol.get("molecule_structures") or {}).get("canonical_smiles")
                if s:
                    return str(s)

        if smiles and str(smiles).strip():
            return str(smiles).strip()
        return None

    def get_full_properties(
        self,
        chembl_id: Optional[str] = None,
        smiles: Optional[str] = None,
    ) -> Optional[PhysChemProperties]:
        """
        ChEMBL 基线 +（若有 canonical SMILES）始终跑 RDKit bundle 与核心合并；
        若 ``_ENRICH_PKA_LOGD`` 为 True，再 pkasolver 补 pKa 及由 logP+pKa 估算 logD。
        """
        chembl_part: Optional[PhysChemProperties] = None
        if chembl_id:
            chembl_part = self.calculate_from_chembl(chembl_id)

        # 这里work是canonical_smiles
        work = self._resolve_working_smiles(chembl_id, smiles, chembl_part)
        if not work:
            if chembl_part:
                return chembl_part
            return None

        logger.info("计算 RDKit feature bundle 并合并核心字段（ChEMBL 优先）")
        try:
            bundle = self.calculate_rdkit_feature_bundle(work)
        except Exception as e:
            logger.warning("RDKit bundle 失败: %s", e)
            if chembl_part:
                return normalize_physchem_numeric_core(
                    self._enrich_pka_logd(chembl_part, work)
                )
            return None

        rd = bundle["rdkit_descriptors"]
        eng = bundle["engineered_features"]

        # 简化：ChEMBL 优先覆盖 RDKit 计算的 pKa/logD（如果开启的话）
        # ro5_violations 和 qed 以 RDKit 计算的为准（已在 rdkit_descriptors 中）
        final_rd = dict(rd)
        if chembl_part:
            if chembl_part.basic_pka and chembl_part.basic_pka.value is not None:
                final_rd["basic_pka"] = chembl_part.basic_pka.value
            if chembl_part.acidic_pka and chembl_part.acidic_pka.value is not None:
                final_rd["acidic_pka"] = chembl_part.acidic_pka.value
            if chembl_part.logd_7_4 and chembl_part.logd_7_4.value is not None:
                final_rd["logd_7_4"] = chembl_part.logd_7_4.value

        # 简化版：核心字段从 rdkit_descriptors 提取（去重后的唯一特征）
        # 保留核心字段是为了向后兼容
        merged = PhysChemProperties(
            mw=SourceInfo(value=final_rd.get("MolWt", 0), source="rdkit"),
            logp=SourceInfo(value=final_rd.get("logP", 0), source="rdkit"),
            logd_7_4=SourceInfo(value=final_rd.get("logd_7_4"), source="rdkit") if final_rd.get("logd_7_4") else None,
            tpsa=SourceInfo(value=final_rd.get("topological_polar_surface_area", 0), source="rdkit"),
            hbd=SourceInfo(value=final_rd.get("NumHDonors", 0), source="rdkit"),
            hba=SourceInfo(value=final_rd.get("NumHAcceptors", 0), source="rdkit"),
            rotatable_bonds=SourceInfo(value=final_rd.get("num_rotatable", 0), source="rdkit"),
            aromatic_rings=SourceInfo(value=final_rd.get("num_aromatic_rings", 0), source="rdkit"),
            heavy_atoms=SourceInfo(value=final_rd.get("HeavyAtomCount", 0), source="rdkit"),
            basic_pka=SourceInfo(value=final_rd.get("basic_pka"), source="rdkit") if final_rd.get("basic_pka") else None,
            acidic_pka=SourceInfo(value=final_rd.get("acidic_pka"), source="rdkit") if final_rd.get("acidic_pka") else None,
            ro5_violations=SourceInfo(value=final_rd.get("ro5_violations", 0), source="rdkit"),
            qed=SourceInfo(value=final_rd.get("qed", 0), source="rdkit"),
            rdkit_descriptors=final_rd,
            engineered_features=eng,
        )
        return normalize_physchem_numeric_core(self._enrich_pka_logd(merged, work))


def create_physchem_calculator() -> PhysChemCalculator:
    return PhysChemCalculator()


def _cli_main() -> None:
    """命令行 / VS Code 调试入口（与 m2_physchem.py 共用）。"""
    import argparse
    import json
    import sys

    project_root = Path(__file__).resolve().parent.parent
    default_out = project_root / "data/output/test_m2.json"

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    ap = argparse.ArgumentParser(
        description="调试 PhysChemCalculator（需 PYTHONPATH 含项目根与 src）"
    )
    ap.add_argument(
        "--mode",
        choices=("full", "chembl", "smiles"),
        default="full",
        help="full: ChEMBL 基线 + RDKit bundle 合并（pKa/logD 富集见 _ENRICH_PKA_LOGD）；"
        "chembl: 仅 ChEMBL 核心；smiles: 仅 RDKit",
    )
    ap.add_argument("--chembl-id", default="CHEMBL25", help="ChEMBL ID（chembl/full）")
    ap.add_argument(
        "--smiles",
        default="CC(=O)Oc1ccccc1C(=O)O",
        help="SMILES（full/smiles）",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=str(default_out),
        help="结果 JSON 保存路径（默认 data/output/test_m2.json）",
    )
    ns = ap.parse_args()
    cid = (ns.chembl_id or "").strip() or None
    smi = (ns.smiles or "").strip() or None

    print(
        "正在计算理化性质（ChEMBL / RDKit；pKa 富集当前关闭）…",
        file=sys.stderr,
        flush=True,
    )
    calc = PhysChemCalculator()
    if ns.mode == "chembl":
        if not cid:
            print("错误: chembl 模式需要有效的 --chembl-id", file=sys.stderr)
            sys.exit(1)
        prop = calc.calculate_from_chembl(cid)
    elif ns.mode == "smiles":
        if not smi:
            print("错误: smiles 模式需要有效的 --smiles", file=sys.stderr)
            sys.exit(1)
        prop = calc.calculate_from_smiles(smi)
    else:
        prop = calc.get_full_properties(chembl_id=cid, smiles=smi)

    if prop is None:
        print("未得到理化性质（返回 None）", file=sys.stderr, flush=True)
        sys.exit(2)

    payload = prop.model_dump()
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text, flush=True)

    out_path = Path(ns.output)
    if not out_path.is_absolute():
        out_path = project_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"已保存 M2 测试结果: {out_path}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    _cli_main()
