"""ChEMBL API 客户端封装（兼容入口，实现位于 data_sources.chembl_client）。"""

from data_sources.chembl_client import (
    CHEMBL_AVAILABLE,
    ChEMBLClient,
    create_chembl_client,
    standard_inchi_key_from_molecule,
)

__all__ = [
    "CHEMBL_AVAILABLE",
    "ChEMBLClient",
    "create_chembl_client",
    "standard_inchi_key_from_molecule",
]
