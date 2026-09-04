"""Versioned controller/worker protocol for ``xvram-torch-bench``.

The protocol is deliberately tiny: every frame is a fixed header followed by a
UTF-8 JSON object.  Keeping the framing here (instead of using pickle or an
unbounded line protocol) makes malformed worker output safe to reject.
"""

from __future__ import annotations

import enum
import json
import struct
from dataclasses import dataclass
from typing import Any, BinaryIO, Mapping


MAGIC_V1 = b"XVT1"
MAGIC_V2 = b"XVT2"
# Backward-compatible public aliases.  Calls which do not opt into v2 must
# continue to emit the byte-identical Phase 4b framing contract.
MAGIC = MAGIC_V1
PROTOCOL_VERSION = 1
MAX_PAYLOAD_BYTES = 1024 * 1024
MAX_TRACE_BATCH_BYTES = 64 * 1024
_HEADER = struct.Struct("!4sBBI")


class ProtocolError(RuntimeError):
    """Raised for invalid, truncated, or oversized protocol frames."""


class MessageType(enum.IntEnum):
    PLAN = 1
    HEARTBEAT = 2
    PROGRESS = 3
    TRACE = 4
    FINAL = 5


@dataclass(frozen=True)
class Frame:
    message_type: MessageType
    payload: dict[str, Any]
    protocol_version: int = 1


def _magic_for(protocol_version: int) -> bytes:
    if protocol_version == 1:
        return MAGIC_V1
    if protocol_version == 2:
        return MAGIC_V2
    raise ProtocolError(f"unsupported XVT protocol version {protocol_version}")


def encode_frame(
    message_type: MessageType,
    payload: Mapping[str, Any],
    *,
    protocol_version: int = 1,
) -> bytes:
    data = json.dumps(
        dict(payload), ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    limit = MAX_TRACE_BATCH_BYTES if message_type is MessageType.TRACE else MAX_PAYLOAD_BYTES
    if len(data) > limit:
        raise ProtocolError(f"{message_type.name.lower()} payload exceeds {limit} bytes")
    return _HEADER.pack(
        _magic_for(protocol_version), protocol_version, int(message_type), len(data)
    ) + data


def write_frame(
    stream: BinaryIO,
    message_type: MessageType,
    payload: Mapping[str, Any],
    *,
    protocol_version: int = 1,
) -> None:
    stream.write(
        encode_frame(message_type, payload, protocol_version=protocol_version)
    )
    stream.flush()


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise ProtocolError("truncated XVT frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(stream: BinaryIO, *, expected_protocol_version: int | None = None) -> Frame:
    header = _read_exact(stream, _HEADER.size)
    magic, version, raw_type, length = _HEADER.unpack(header)
    if magic == MAGIC_V1:
        protocol_version = 1
    elif magic == MAGIC_V2:
        protocol_version = 2
    else:
        raise ProtocolError("invalid XVT magic")
    if (
        expected_protocol_version is not None
        and protocol_version != expected_protocol_version
    ):
        raise ProtocolError(
            "unexpected XVT{} frame in XVT{} stream".format(
                protocol_version, expected_protocol_version
            )
        )
    if version != protocol_version:
        raise ProtocolError(f"unsupported XVT{protocol_version} version {version}")
    try:
        message_type = MessageType(raw_type)
    except ValueError as exc:
        raise ProtocolError(
            f"unknown XVT{protocol_version} message type {raw_type}"
        ) from exc
    limit = MAX_TRACE_BATCH_BYTES if message_type is MessageType.TRACE else MAX_PAYLOAD_BYTES
    if length > limit:
        raise ProtocolError(f"{message_type.name.lower()} payload exceeds {limit} bytes")
    raw_payload = _read_exact(stream, length)
    try:
        payload = json.loads(raw_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(
            f"XVT{protocol_version} payload is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"XVT{protocol_version} payload must be a JSON object")
    return Frame(
        message_type=message_type,
        payload=payload,
        protocol_version=protocol_version,
    )
