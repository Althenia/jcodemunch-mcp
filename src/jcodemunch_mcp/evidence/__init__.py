"""Compile-time evidence ingestion (SCIP).

The compile-time sibling of the ``runtime/`` trace-ingestion package:
``import-scip`` parses a SCIP index file (the protobuf artifact emitted by
scip-typescript, scip-python, scip-java, scip-go, rust-analyzer, scip-clang)
and stores compiler-verified reference/implementation edges in the repo's
``scip_*`` tables. Read-only with respect to the user's code; the only write
is to the per-repo index database.
"""

from .scip import (
    ScipDocument,
    ScipIndex,
    ScipOccurrence,
    ScipRelationship,
    ScipSymbolInfo,
    display_name_from_symbol,
    is_local_symbol,
    parse_scip_bytes,
    parse_scip_file,
)
from .scip_ingest import ingest_scip_file

__all__ = [
    "ScipDocument",
    "ScipIndex",
    "ScipOccurrence",
    "ScipRelationship",
    "ScipSymbolInfo",
    "display_name_from_symbol",
    "ingest_scip_file",
    "is_local_symbol",
    "parse_scip_bytes",
    "parse_scip_file",
]
