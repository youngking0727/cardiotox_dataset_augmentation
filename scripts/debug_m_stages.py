"""调试入口：单独跑 M2、M3（A/B/C 或汇总）、M4–M6（与 .vscode/launch.json 配套）。

用法:
  PYTHONPATH=src python scripts/debug_m_stages.py --stage m3a --chembl-id CHEMBL25
  PYTHONPATH=src python scripts/debug_m_stages.py --stage m2 --chembl-id CHEMBL473 --smiles '...'
  PYTHONPATH=src python scripts/debug_m_stages.py --stage m3 --chembl-id CHEMBL473 --drug-name dofetilide
  PYTHONPATH=src python scripts/debug_m_stages.py --stage m4 --chembl-id CHEMBL473 --smiles '...' \\
      --m2-json data/output/test_m2.json --m3-json data/output/test_m3.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

_SRC = Path(__file__).resolve().parent.parent / "src"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _json_print(obj: object) -> None:
    if hasattr(obj, "model_dump"):
        data = obj.model_dump(mode="json")
    elif isinstance(obj, dict):
        data = {
            k: v.model_dump(mode="json") if hasattr(v, "model_dump") else v
            for k, v in obj.items()
        }
    else:
        data = obj
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def _resolve_project_path(p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return _PROJECT_ROOT / path


def _load_pipeline_config() -> Dict[str, Any]:
    import yaml

    cfg_rel = os.environ.get("PIPELINE_CONFIG", "config/pipeline.yaml")
    cfg_path = Path(cfg_rel) if Path(cfg_rel).is_absolute() else _PROJECT_ROOT / cfg_rel
    if not cfg_path.is_file():
        return {}
    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return raw.get("pipeline", raw)


def _literature_retriever_from_config():
    """与 M3C / run_pipeline 一致：读 pipeline.yaml 构造 PubMedClient。"""
    from m3c_literature import LiteratureRetriever
    from utils.chembl_client import ChEMBLClient
    from utils.pubmed_client import PubMedClient, apply_ncbi_settings_from_config

    config = _load_pipeline_config()
    if not config:
        print(
            "未找到 config/pipeline.yaml，PubMed 将仅使用环境变量 NCBI_EMAIL / NCBI_API_KEY",
            file=sys.stderr,
            flush=True,
        )
    else:
        apply_ncbi_settings_from_config(config)
    pub_cfg = (config.get("api") or {}).get("pubmed") or {}
    pubmed_client = PubMedClient(
        rate_limit=float(pub_cfg.get("rate_limit", 0.5)),
        ncbi_email=pub_cfg.get("ncbi_email"),
        ncbi_api_key=pub_cfg.get("ncbi_api_key"),
        ncbi_tool=pub_cfg.get("ncbi_tool"),
    )
    return LiteratureRetriever(
        chembl_client=ChEMBLClient(),
        pubmed_client=pubmed_client,
    )


def _run_m2(
    chembl_id: str,
    smiles: str,
    *,
    mode: str,
    output: Path,
) -> None:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    from m2_physchem_calculator import PhysChemCalculator

    cid = (chembl_id or "").strip() or None
    smi = (smiles or "").strip() or None
    calc = PhysChemCalculator()
    if mode == "chembl":
        if not cid:
            print("错误: chembl 模式需要有效的 --chembl-id", file=sys.stderr)
            sys.exit(1)
        prop = calc.calculate_from_chembl(cid)
    elif mode == "smiles":
        if not smi:
            print("错误: smiles 模式需要有效的 --smiles", file=sys.stderr)
            sys.exit(1)
        prop = calc.calculate_from_smiles(smi)
    else:
        prop = calc.get_full_properties(chembl_id=cid, smiles=smi)

    if prop is None:
        print("未得到理化性质（返回 None）", file=sys.stderr, flush=True)
        sys.exit(2)

    out_path = _resolve_project_path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(prop.model_dump(), ensure_ascii=False, indent=2)
    out_path.write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"已保存 M2 结果: {out_path}", file=sys.stderr, flush=True)


def _run_m3_export(
    chembl_id: str,
    drug_name: str,
    max_pubmed: int,
    *,
    refresh_clinical: bool,
    output: Path,
) -> None:
    """串联 M3A/M3B/M3C，写入 test_m3.json（bioactivity + clinical + literature）。"""
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    from m3a_bioactivity import BioactivityRetriever
    from m3b_clinical import ClinicalStatusRetriever
    from schemas import ClinicalEvidence, LiteratureEvidence

    bio_r = BioactivityRetriever()
    bio: Dict[str, Any] = bio_r.retrieve(chembl_id)
    bio_dump = {k: v.model_dump(mode="json") for k, v in bio.items()}

    clin_r = ClinicalStatusRetriever()
    clinical: ClinicalEvidence = clin_r.retrieve(
        chembl_id,
        drug_name=drug_name,
        refresh_chembl_clinical=refresh_clinical,
    )

    lit_r = _literature_retriever_from_config()
    literature: LiteratureEvidence = lit_r.retrieve(
        chembl_id, drug_name=drug_name, max_pubmed=max_pubmed
    )

    payload = {
        "chembl_id": chembl_id,
        "drug_name": drug_name,
        "bioactivity": bio_dump,
        "clinical": clinical.model_dump(mode="json"),
        "literature": literature.model_dump(mode="json"),
    }
    out_path = _resolve_project_path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    out_path.write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"已保存 M3 汇总: {out_path}", file=sys.stderr, flush=True)


def _run_m3a(chembl_id: str) -> None:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    from m3a_bioactivity import BioactivityRetriever

    r = BioactivityRetriever()
    out = r.retrieve(chembl_id)
    _json_print(out)


def _run_m3b(
    chembl_id: str,
    drug_name: str | None,
    *,
    refresh_clinical: bool,
) -> None:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    from m3b_clinical import ClinicalStatusRetriever

    r = ClinicalStatusRetriever()
    ev = r.retrieve(
        chembl_id,
        drug_name=drug_name,
        refresh_chembl_clinical=refresh_clinical,
    )
    _json_print(ev)


def _run_m3c(chembl_id: str, drug_name: str, max_pubmed: int) -> None:
    """
    M3C：与 run_pipeline 一致，从 config/pipeline.yaml 的 api.pubmed 构造 PubMedClient。
    不依赖 ``import run_pipeline``（避免模块加载顺序问题），且显式传入 client，不只靠 NCBI_EMAIL 环境变量。
    """
    r = _literature_retriever_from_config()
    ev = r.retrieve(chembl_id, drug_name=drug_name, max_pubmed=max_pubmed)
    _json_print(ev)


def _run_m4(
    chembl_id: str,
    smiles: str,
    *,
    m2_json: Path | None,
    m3_json: Path | None,
) -> None:
    from m4_assembler import EvidenceBundleAssembler
    from schemas import (
        ClinicalEvidence,
        LiteratureEvidence,
        PhysChemProperties,
        TargetBioactivity,
    )

    physchem_properties = None
    bioactivity_evidence = None
    clinical_evidence = None
    literature_evidence = None

    if m2_json is not None and m2_json.is_file():
        raw = json.loads(m2_json.read_text(encoding="utf-8"))
        physchem_properties = PhysChemProperties.model_validate(raw)
    else:
        p = m2_json or Path("(未指定)")
        print(f"警告: 未找到或未传入 M2 JSON（{p}），理化性质将为空占位", file=sys.stderr, flush=True)

    if m3_json is not None and m3_json.is_file():
        raw3 = json.loads(m3_json.read_text(encoding="utf-8"))
        bio_raw = raw3.get("bioactivity") or {}
        bioactivity_evidence = {
            k: TargetBioactivity.model_validate(v) for k, v in bio_raw.items()
        }
        clinical_evidence = ClinicalEvidence.model_validate(raw3.get("clinical") or {})
        literature_evidence = LiteratureEvidence.model_validate(raw3.get("literature") or {})
    else:
        p = m3_json or Path("(未指定)")
        print(f"警告: 未找到或未传入 M3 JSON（{p}），活性/临床/文献将为空占位", file=sys.stderr, flush=True)

    a = EvidenceBundleAssembler()
    bundle = a.assemble(
        chembl_id=chembl_id,
        smiles=smiles,
        source_path="debug/smoke",
        label_value=None,
        label_source="undetermined",
        physchem_properties=physchem_properties,
        bioactivity_evidence=bioactivity_evidence,
        clinical_evidence=clinical_evidence,
        literature_evidence=literature_evidence,
    )
    _json_print(bundle)


def _run_m5() -> None:
    from m5_sufficiency import EvidenceSufficiencyJudge
    from schemas import (
        BioactivityMeasurement,
        ClinicalEvidence,
        ExternalDBFlags,
        LiteratureEvidence,
        PubMedArticle,
        TargetBioactivity,
        WithdrawalInfo,
    )

    print(
        "（demo）内置合成样例，仅验证分层规则可运行；不代表任何真实分子的证据包或最终标签。",
        file=sys.stderr,
        flush=True,
    )
    judge = EvidenceSufficiencyJudge()
    herg = TargetBioactivity(
        target="hERG",
        target_chembl_id="CHEMBL240",
        measurements=[
            BioactivityMeasurement(type="IC50", value=1.2, units="uM", pchembl=5.92)
        ],
    )
    clinical = ClinicalEvidence(
        max_phase=4,
        approved=True,
        withdrawn=WithdrawalInfo(flag=True, reason="QT prolongation"),
        black_box_warning=True,
        clinical_trials=[],
        external_db_flags={
            "cardiotox": ExternalDBFlags(present=True, risk_level="high"),
            "etox": ExternalDBFlags(present=False),
            "dilirank": ExternalDBFlags(present=False),
        },
        fda_label_warnings=[],
    )
    literature = LiteratureEvidence(
        pubmed_articles=[
            PubMedArticle(
                pmid="12345678",
                title="hERG and QT",
                mesh_terms=[],
                is_review=False,
                relevance_keywords_hit=["hERG", "QT prolongation"],
                molecule_mentioned=True,
                publication_year=2020,
            )
        ],
        patents=[],
    )
    density = judge.compute_evidence_density(
        bioactivity_evidence={"hERG": herg},
        clinical_evidence=clinical,
        literature_evidence=literature,
    )
    detail = judge.judge_label_detailed(
        bioactivity_evidence={"hERG": herg}, clinical_evidence=clinical
    )
    print("--- evidence_density (第一层：充分性 total_score → sufficiency_tier) ---")
    _json_print(density)
    print("--- judge_label_detailed (第二层：标签，与充分性解耦) ---")
    _json_print(detail)
    print("--- judge_label (兼容字段 label, confidence) ---")
    print(
        json.dumps(
            {"label": detail.label, "confidence": detail.confidence},
            ensure_ascii=False,
            indent=2,
        )
    )


def _run_m6() -> None:
    from m6_conflict import ConflictDetector
    from schemas import (
        BioactivityMeasurement,
        ClinicalEvidence,
        ExternalDBFlags,
        LiteratureEvidence,
        PubMedArticle,
        TargetBioactivity,
        WithdrawalInfo,
    )

    print(
        "（demo）内置合成样例：故意构造「强 hERG 体外 + 无临床警告」以展示 A vs B 冲突；"
        "不代表真实分子或管线数据。",
        file=sys.stderr,
        flush=True,
    )
    detector = ConflictDetector()
    herg = TargetBioactivity(
        target="hERG",
        target_chembl_id="CHEMBL240",
        measurements=[
            BioactivityMeasurement(type="IC50", value=0.5, units="uM")
        ],
    )
    clinical = ClinicalEvidence(
        max_phase=4,
        approved=True,
        withdrawn=WithdrawalInfo(flag=False),
        black_box_warning=None,
        clinical_trials=[],
        external_db_flags={
            "cardiotox": ExternalDBFlags(present=False),
            "etox": ExternalDBFlags(present=False),
            "dilirank": ExternalDBFlags(present=False),
        },
        fda_label_warnings=[],
    )
    literature = LiteratureEvidence(
        pubmed_articles=[
            PubMedArticle(
                pmid="1",
                title="x",
                mesh_terms=[],
                is_review=False,
                relevance_keywords_hit=[],
                molecule_mentioned=False,
                publication_year=2020,
            )
        ],
        patents=[],
    )
    result = detector.detect(
        bioactivity_evidence={"hERG": herg},
        clinical_evidence=clinical,
        literature_evidence=literature,
    )
    out = {
        "has_conflict": result.has_conflict,
        "conflict_A_vs_B": result.conflict_A_vs_B,
        "conflict_within_C": result.conflict_within_C,
        "conflict_details": result.conflict_details,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


def main() -> None:
    ap = argparse.ArgumentParser(description="调试 M2、M3、M4–M6 单阶段")
    ap.add_argument(
        "--stage",
        choices=("m2", "m3", "m3a", "m3b", "m3c", "m4", "m5", "m6"),
        required=True,
    )
    ap.add_argument("--chembl-id", default="CHEMBL25", help="ChEMBL ID（M2/M3/M4）")
    ap.add_argument(
        "--smiles",
        default="CC(=O)Oc1ccccc1C(=O)O",
        help="M2 full/smiles 与 M4 组装用 SMILES",
    )
    ap.add_argument(
        "--mode",
        choices=("full", "chembl", "smiles"),
        default="full",
        help="M2：理化计算模式（与 m2_physchem_calculator.py 一致）",
    )
    ap.add_argument(
        "--m2-output",
        default="data/output/test_m2.json",
        help="M2：写出路径（默认 data/output/test_m2.json）",
    )
    ap.add_argument(
        "--m3-output",
        default="data/output/test_m3.json",
        help="M3 汇总：写出路径（默认 data/output/test_m3.json）",
    )
    ap.add_argument(
        "--m2-json",
        default="data/output/test_m2.json",
        help="M4：从该文件读入 PhysChemProperties（默认 data/output/test_m2.json）",
    )
    ap.add_argument(
        "--m3-json",
        default="data/output/test_m3.json",
        help="M4：从该文件读入 bioactivity/clinical/literature（默认 data/output/test_m3.json）",
    )
    ap.add_argument(
        "--drug-name",
        default=None,
        help="M3/M3B/M3C 药物名（ClinicalTrials/PubMed）；M3 默认用 ChEMBL pref_name",
    )
    ap.add_argument("--max-pubmed", type=int, default=5, help="M3/M3C PubMed 条数上限")
    ap.add_argument(
        "--refresh-clinical",
        action="store_true",
        help="M3/M3B：删除 chembl_clinical_{chembl_id} 缓存后再拉 ChEMBL drug",
    )
    ns = ap.parse_args()

    if ns.stage == "m2":
        _run_m2(
            ns.chembl_id,
            ns.smiles,
            mode=ns.mode,
            output=Path(ns.m2_output),
        )
    elif ns.stage == "m3":
        dn = (ns.drug_name or "").strip()
        if not dn:
            from utils.chembl_client import ChEMBLClient

            mol, ok = ChEMBLClient().get_molecule_by_chembl_id(ns.chembl_id)
            if ok and mol and isinstance(mol, dict):
                dn = (str(mol.get("pref_name") or "")).strip()
            if not dn:
                dn = ns.chembl_id
        _run_m3_export(
            ns.chembl_id,
            dn,
            ns.max_pubmed,
            refresh_clinical=ns.refresh_clinical,
            output=Path(ns.m3_output),
        )
    elif ns.stage == "m3a":
        _run_m3a(ns.chembl_id)
    elif ns.stage == "m3b":
        _run_m3b(ns.chembl_id, ns.drug_name, refresh_clinical=ns.refresh_clinical)
    elif ns.stage == "m3c":
        dn = ns.drug_name or "aspirin"
        _run_m3c(ns.chembl_id, dn, ns.max_pubmed)
    elif ns.stage == "m4":
        m2p = _resolve_project_path(ns.m2_json)
        m3p = _resolve_project_path(ns.m3_json)
        _run_m4(ns.chembl_id, ns.smiles, m2_json=m2p, m3_json=m3p)
    elif ns.stage == "m5":
        _run_m5()
    else:
        _run_m6()


if __name__ == "__main__":
    main()
