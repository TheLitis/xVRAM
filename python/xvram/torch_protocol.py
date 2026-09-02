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


MAGIC = b"XVT1"
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


def encode_frame(message_type: MessageType, payload: Mapping[str, Any]) -> bytes:
    data = json.dumps(
        dict(payload), ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    limit = MAX_TRACE_BATCH_BYTES if message_type is MessageType.TRACE else MAX_PAYLOAD_BYTES
    if len(data) > limit:
        raise ProtocolError(f"{message_type.name.lower()} payload exceeds {limit} bytes")
    return _HEADER.pack(MAGIC, PROTOCOL_VERSION, int(message_type), len(data)) + data


def write_frame(
    stream: BinaryIO, message_type: MessageType, payload: Mapping[str, Any]
) -> None:
    stream.write(encode_frame(message_type, payload))
    stream.flush()


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise ProtocolError("truncated XVT1 frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(stream: BinaryIO) -> Frame:
    header = _read_exact(stream, _HEADER.size)
    magic, version, raw_type, length = _HEADER.unpack(header)
    if magic != MAGIC:
        raise ProtocolError("invalid XVT1 magic")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported XVT1 version {version}")
    try:
        message_type = MessageType(raw_type)
    except ValueError as exc:
        raise ProtocolError(f"unknown XVT1 message type {raw_type}") from exc
    limit = MAX_TRACE_BATCH_BYTES if message_type is MessageType.TRACE else MAX_PAYLOAD_BYTES
    if length > limit:
        raise ProtocolError(f"{message_type.name.lower()} payload exceeds {limit} bytes")
    raw_payload = _read_exact(stream, length)
    try:
        payload = json.loads(raw_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("XVT1 payload is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("XVT1 payload must be a JSON object")
    return Frame(message_type=message_type, payload=payload)
