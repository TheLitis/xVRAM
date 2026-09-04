from __future__ import annotations

import io
import struct
import unittest

from xvram.torch_protocol import (
    MAGIC,
    MAGIC_V1,
    MAGIC_V2,
    MAX_PAYLOAD_BYTES,
    Frame,
    MessageType,
    ProtocolError,
    encode_frame,
    read_frame,
)


class TorchProtocolTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        encoded = encode_frame(MessageType.PROGRESS, {"operation": 7, "ok": True})
        self.assertEqual(MAGIC, MAGIC_V1)
        self.assertEqual(encoded[:4], MAGIC_V1)
        frame = read_frame(io.BytesIO(encoded))
        self.assertEqual(frame.message_type, MessageType.PROGRESS)
        self.assertEqual(frame.payload, {"operation": 7, "ok": True})
        self.assertEqual(frame.protocol_version, 1)

    def test_compression_protocol_uses_xvt2_without_changing_v1_default(self) -> None:
        encoded = encode_frame(
            MessageType.PROGRESS,
            {"operation": 8, "compression": "adaptive"},
            protocol_version=2,
        )
        self.assertEqual(encoded[:4], MAGIC_V2)
        self.assertEqual(encoded[4], 2)
        frame = read_frame(io.BytesIO(encoded), expected_protocol_version=2)
        self.assertEqual(frame.protocol_version, 2)
        self.assertEqual(frame.payload["compression"], "adaptive")

        with self.assertRaisesRegex(ProtocolError, "unexpected XVT2 frame"):
            read_frame(io.BytesIO(encoded), expected_protocol_version=1)

        default = encode_frame(MessageType.HEARTBEAT, {})
        self.assertEqual(default[:4], MAGIC_V1)
        self.assertEqual(default[4], 1)

    def test_truncation_is_rejected(self) -> None:
        encoded = encode_frame(MessageType.FINAL, {"report": {}})
        with self.assertRaisesRegex(ProtocolError, "truncated"):
            read_frame(io.BytesIO(encoded[:-1]))

    def test_oversized_payload_is_rejected_before_read(self) -> None:
        header = struct.pack("!4sBBI", MAGIC, 1, int(MessageType.FINAL), MAX_PAYLOAD_BYTES + 1)
        with self.assertRaisesRegex(ProtocolError, "exceeds"):
            read_frame(io.BytesIO(header))

    def test_trace_has_smaller_bound(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "trace payload"):
            encode_frame(MessageType.TRACE, {"data": "x" * (65 * 1024)})

    def test_non_object_payload_is_rejected(self) -> None:
        raw = b"[]"
        header = struct.pack("!4sBBI", MAGIC, 1, int(MessageType.PLAN), len(raw))
        with self.assertRaisesRegex(ProtocolError, "object"):
            read_frame(io.BytesIO(header + raw))


if __name__ == "__main__":
    unittest.main()
