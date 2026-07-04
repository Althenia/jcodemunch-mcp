"""Minimal SCIP index reader — hand-rolled protobuf wire-format walker.

SCIP (Sourcegraph Code Intelligence Protocol) indexes are a single
serialized ``scip.Index`` protobuf message. We only need a small, stable
subset of its fields, so rather than adding a ``protobuf`` dependency plus
a generated ``scip_pb2.py`` to maintain, this module walks the protobuf
wire format directly: varints and length-delimited fields. Unknown fields
are skipped by construction, so additions to the SCIP schema are ignored
rather than fatal. (Precedent: the hand-rolled minimal TOML/YAML readers
in ``tools/list_workspaces.py``.)

Field numbers below are from scip.proto and are frozen by protobuf's
compatibility rules:

    Index:             metadata=1, documents=2, external_symbols=3
    Metadata:          tool_info=2, project_root=3
    ToolInfo:          name=1, version=2
    Document:          relative_path=1, occurrences=2, symbols=3, language=4
    Occurrence:        range=1 (packed int32), symbol=2, symbol_roles=3
    SymbolInformation: symbol=1, relationships=4
    Relationship:      symbol=1, is_reference=2, is_implementation=3,
                       is_type_definition=4

``Occurrence.range`` is 0-based ``[startLine, startChar, endLine, endChar]``
or the 3-int same-line form ``[startLine, startChar, endChar]``; both are
handled (we only consume the start line, converted to 1-based).
"""

import gzip
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Union

logger = logging.getLogger(__name__)

# Occurrence.symbol_roles bitmask (scip.proto SymbolRole)
SYMBOL_ROLE_DEFINITION = 0x1
SYMBOL_ROLE_IMPORT = 0x2

_GZIP_MAGIC = b"\x1f\x8b"


# ── Wire-format primitives ──────────────────────────────────────────────


def _read_varint(buf: Union[bytes, memoryview], pos: int) -> tuple[int, int]:
    """Read one base-128 varint at ``pos``. Returns (value, next_pos)."""
    result = 0
    shift = 0
    end = len(buf)
    while True:
        if pos >= end:
            raise ValueError("truncated varint")
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")


def _iter_fields(buf: Union[bytes, memoryview]) -> Iterator[tuple[int, int, Union[int, memoryview]]]:
    """Yield ``(field_number, wire_type, payload)`` for one message buffer.

    Payload is the varint value for wire type 0 and a memoryview slice for
    wire types 1/2/5. Deprecated group wire types (3/4) are rejected —
    SCIP never uses them, so hitting one means the input is not SCIP.
    """
    view = memoryview(buf)
    pos = 0
    end = len(view)
    while pos < end:
        tag, pos = _read_varint(view, pos)
        field_no = tag >> 3
        wire_type = tag & 0x7
        if field_no == 0:
            raise ValueError("invalid field number 0")
        if wire_type == 0:
            value, pos = _read_varint(view, pos)
            yield field_no, wire_type, value
        elif wire_type == 2:
            length, pos = _read_varint(view, pos)
            if pos + length > end:
                raise ValueError("truncated length-delimited field")
            yield field_no, wire_type, view[pos:pos + length]
            pos += length
        elif wire_type == 1:
            if pos + 8 > end:
                raise ValueError("truncated fixed64 field")
            yield field_no, wire_type, view[pos:pos + 8]
            pos += 8
        elif wire_type == 5:
            if pos + 4 > end:
                raise ValueError("truncated fixed32 field")
            yield field_no, wire_type, view[pos:pos + 4]
            pos += 4
        else:
            raise ValueError(f"unsupported wire type {wire_type}")


def _extend_int32s(target: list[int], wire_type: int, payload: Union[int, memoryview]) -> None:
    """Append packed (wire 2) or single (wire 0) repeated int32 values."""
    if wire_type == 0:
        target.append(int(payload))  # type: ignore[arg-type]
        return
    if wire_type == 2:
        view = payload  # type: ignore[assignment]
        pos = 0
        end = len(view)  # type: ignore[arg-type]
        while pos < end:
            value, pos = _read_varint(view, pos)  # type: ignore[arg-type]
            target.append(value)


def _utf8(payload: Union[int, memoryview]) -> str:
    if isinstance(payload, int):
        return ""
    return bytes(payload).decode("utf-8", errors="replace")


# ── Parsed shapes ───────────────────────────────────────────────────────


@dataclass
class ScipOccurrence:
    range: tuple[int, ...] = ()
    symbol: str = ""
    symbol_roles: int = 0

    @property
    def start_line(self) -> Optional[int]:
        """1-based start line (SCIP ranges are 0-based)."""
        if self.range:
            return self.range[0] + 1
        return None


@dataclass
class ScipRelationship:
    symbol: str = ""
    is_reference: bool = False
    is_implementation: bool = False
    is_type_definition: bool = False


@dataclass
class ScipSymbolInfo:
    symbol: str = ""
    relationships: list[ScipRelationship] = field(default_factory=list)


@dataclass
class ScipDocument:
    relative_path: str = ""
    language: str = ""
    occurrences: list[ScipOccurrence] = field(default_factory=list)
    symbols: list[ScipSymbolInfo] = field(default_factory=list)


@dataclass
class ScipIndex:
    tool_name: str = ""
    tool_version: str = ""
    project_root: str = ""
    documents: list[ScipDocument] = field(default_factory=list)
    external_symbols: list[ScipSymbolInfo] = field(default_factory=list)


# ── Message parsers ─────────────────────────────────────────────────────


def _parse_occurrence(buf: memoryview) -> ScipOccurrence:
    occ = ScipOccurrence()
    range_values: list[int] = []
    for field_no, wire_type, payload in _iter_fields(buf):
        if field_no == 1:
            _extend_int32s(range_values, wire_type, payload)
        elif field_no == 2:
            occ.symbol = _utf8(payload)
        elif field_no == 3 and wire_type == 0:
            occ.symbol_roles = int(payload)  # type: ignore[arg-type]
    occ.range = tuple(range_values)
    return occ


def _parse_relationship(buf: memoryview) -> ScipRelationship:
    rel = ScipRelationship()
    for field_no, wire_type, payload in _iter_fields(buf):
        if field_no == 1:
            rel.symbol = _utf8(payload)
        elif field_no == 2 and wire_type == 0:
            rel.is_reference = bool(payload)
        elif field_no == 3 and wire_type == 0:
            rel.is_implementation = bool(payload)
        elif field_no == 4 and wire_type == 0:
            rel.is_type_definition = bool(payload)
    return rel


def _parse_symbol_info(buf: memoryview) -> ScipSymbolInfo:
    info = ScipSymbolInfo()
    for field_no, wire_type, payload in _iter_fields(buf):
        if field_no == 1:
            info.symbol = _utf8(payload)
        elif field_no == 4 and wire_type == 2:
            info.relationships.append(_parse_relationship(payload))  # type: ignore[arg-type]
    return info


def _parse_document(buf: memoryview) -> ScipDocument:
    doc = ScipDocument()
    for field_no, wire_type, payload in _iter_fields(buf):
        if field_no == 1:
            doc.relative_path = _utf8(payload)
        elif field_no == 2 and wire_type == 2:
            doc.occurrences.append(_parse_occurrence(payload))  # type: ignore[arg-type]
        elif field_no == 3 and wire_type == 2:
            doc.symbols.append(_parse_symbol_info(payload))  # type: ignore[arg-type]
        elif field_no == 4:
            doc.language = _utf8(payload)
    return doc


def _parse_metadata(buf: memoryview, index: ScipIndex) -> None:
    for field_no, wire_type, payload in _iter_fields(buf):
        if field_no == 2 and wire_type == 2:
            for sub_no, _sub_wire, sub_payload in _iter_fields(payload):  # type: ignore[arg-type]
                if sub_no == 1:
                    index.tool_name = _utf8(sub_payload)
                elif sub_no == 2:
                    index.tool_version = _utf8(sub_payload)
        elif field_no == 3:
            index.project_root = _utf8(payload)


def parse_scip_bytes(data: bytes) -> ScipIndex:
    """Parse a serialized ``scip.Index`` message.

    Raises:
        ValueError: when the buffer is not parseable as protobuf or parses
            but contains neither metadata nor documents (i.e., it is
            structurally valid protobuf that is clearly not a SCIP index).
    """
    index = ScipIndex()
    try:
        for field_no, wire_type, payload in _iter_fields(data):
            if field_no == 1 and wire_type == 2:
                _parse_metadata(payload, index)  # type: ignore[arg-type]
            elif field_no == 2 and wire_type == 2:
                index.documents.append(_parse_document(payload))  # type: ignore[arg-type]
            elif field_no == 3 and wire_type == 2:
                index.external_symbols.append(_parse_symbol_info(payload))  # type: ignore[arg-type]
    except ValueError as e:
        raise ValueError(
            f"not a SCIP index ({e}). Expected the protobuf artifact produced by "
            "scip-typescript / scip-python / scip-java / scip-go / rust-analyzer."
        ) from e
    if not index.documents and not index.tool_name:
        raise ValueError(
            "not a SCIP index (no metadata or documents found). Expected the "
            "protobuf artifact produced by scip-typescript / scip-python / "
            "scip-java / scip-go / rust-analyzer."
        )
    return index


def parse_scip_file(path: str) -> ScipIndex:
    """Read and parse a ``.scip`` file; ``.gz``-compressed input is handled
    transparently (magic-byte sniff, not extension)."""
    raw = Path(path).read_bytes()
    if raw[:2] == _GZIP_MAGIC:
        raw = gzip.decompress(raw)
    return parse_scip_bytes(raw)


# ── Symbol-string helpers ───────────────────────────────────────────────


def is_local_symbol(symbol: str) -> bool:
    """SCIP file-local symbols (``local N``) — function internals, skipped
    for cross-symbol edges."""
    return symbol.startswith("local ")


def display_name_from_symbol(symbol: str) -> Optional[str]:
    """Best-effort local name from a SCIP symbol string.

    A SCIP symbol looks like
    ``scip-typescript npm pkg 1.0.0 src/`file.ts`/Class#method().`` —
    scheme, package manager, package name, version, then descriptors.
    We extract the last descriptor's name as a resolution *fallback* only;
    the primary resolution channel is (file, line), so a miss here is
    tolerated rather than fatal.
    """
    if not symbol or is_local_symbol(symbol):
        return None
    parts = symbol.split(" ", 4)
    descriptor = parts[4] if len(parts) == 5 else parts[-1]
    d = descriptor.rstrip(".")
    if d.endswith("()"):
        d = d[:-2]
    d = d.rstrip("#/:!.")
    if not d:
        return None
    # Backtick-escaped final segment (names containing separator chars)
    if d.endswith("`"):
        start = d.rfind("`", 0, len(d) - 1)
        if start != -1:
            inner = d[start + 1:-1]
            return inner or None
    for i in range(len(d) - 1, -1, -1):
        if d[i] in "/#.(":
            tail = d[i + 1:]
            return tail or None
    return d
