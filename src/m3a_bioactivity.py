"""M3-A: 靶点活性检索器模块（兼容入口，实现见 m3a_bioactivity_retriever）。"""

from m3a_bioactivity_retriever import (
    DEFAULT_TARGETS,
    BioactivityRetriever,
    create_bioactivity_retriever,
)

__all__ = [
    "DEFAULT_TARGETS",
    "BioactivityRetriever",
    "create_bioactivity_retriever",
]
