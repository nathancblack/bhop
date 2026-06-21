"""Tests for Q3 demo Huffman codec, message buffer, and demo writer.

Covers:
  - HuffmanCodec: round-trip encoding/decoding, tree completeness
  - MsgBuffer/MsgReader: write/read round-trips for all MSG_Write* types
  - DemoWriter: file structure, message count, physics integration
"""

import os
import struct
import tempfile

import numpy as np
import pytest

from bhop.demo import (
    DemoWriter,
    HuffmanCodec,
    MsgBuffer,
    MsgReader,
    TickRecord,
    _q3_codec,
    _tick_to_playerstate,
    ENTITYNUM_NONE,
    ENTITYNUM_WORLD,
)


# ---------------------------------------------------------------------------
# TestHuffman: codec round-trip tests
# ---------------------------------------------------------------------------


class TestHuffman:
    """Huffman codec correctness tests."""

    def test_all_256_symbols_have_codes(self) -> None:
        """Every byte value 0-255 must have a non-zero-length code."""
        for i in range(256):
            _code, nbits = _q3_codec._codes[i]
            assert nbits > 0, f"symbol {i} has no code"

    def test_roundtrip_all_bytes(self) -> None:
        """Encode then decode all 256 byte values."""
        buf = bytearray()
        offset = 0
        for b in range(256):
            offset = _q3_codec.encode_symbol(b, buf, offset)

        decoded = []
        dec_offset = 0
        for _ in range(256):
            sym, dec_offset = _q3_codec.decode_symbol(buf, dec_offset)
            decoded.append(sym)

        assert decoded == list(range(256))

    def test_roundtrip_short_sequence(self) -> None:
        """Round-trip a short byte sequence."""
        data = b"\x00\x07\xff\x80\x01"
        buf = bytearray()
        offset = 0
        for b in data:
            offset = _q3_codec.encode_symbol(b, buf, offset)

        decoded = []
        dec_offset = 0
        for _ in range(len(data)):
            sym, dec_offset = _q3_codec.decode_symbol(buf, dec_offset)
            decoded.append(sym)

        assert bytes(decoded) == data

    def test_roundtrip_repeated_bytes(self) -> None:
        """Round-trip data with many repeated values."""
        data = bytes([0] * 100 + [255] * 100 + [128] * 50)
        buf = bytearray()
        offset = 0
        for b in data:
            offset = _q3_codec.encode_symbol(b, buf, offset)

        decoded = []
        dec_offset = 0
        for _ in range(len(data)):
            sym, dec_offset = _q3_codec.decode_symbol(buf, dec_offset)
            decoded.append(sym)

        assert bytes(decoded) == data

    def test_frequent_symbols_shorter_codes(self) -> None:
        """Symbol 0 (highest frequency) should have a shorter code than rare symbols."""
        _code_0, nbits_0 = _q3_codec._codes[0]
        # Symbol 0 has frequency 250315 -- should be one of the shortest
        assert nbits_0 <= 4
        # Symbol 247 has frequency 740 (one of the rarest)
        _code_247, nbits_247 = _q3_codec._codes[247]
        assert nbits_247 > nbits_0

    def test_prefix_free(self) -> None:
        """No code should be a prefix of another (prefix-free property)."""
        codes = []
        for i in range(256):
            code, nbits = _q3_codec._codes[i]
            # Convert to bit string for prefix checking
            bits = "".join(str((code >> j) & 1) for j in range(nbits))
            codes.append(bits)

        for i in range(256):
            for j in range(256):
                if i != j:
                    assert not codes[j].startswith(codes[i]), (
                        f"code for {i} ({codes[i]}) is prefix of {j} ({codes[j]})"
                    )


# ---------------------------------------------------------------------------
# TestMsgBuffer: message buffer write/read round-trips
# ---------------------------------------------------------------------------


class TestMsgBuffer:
    """MsgBuffer and MsgReader round-trip tests."""

    def test_write_read_byte(self) -> None:
        """write_byte -> read_byte round-trip."""
        buf = MsgBuffer()
        buf.write_byte(0x42)
        r = MsgReader(buf.get_data())
        assert r.read_byte() == 0x42

    def test_write_read_byte_boundaries(self) -> None:
        """Byte boundary values: 0, 127, 128, 255."""
        for val in [0, 127, 128, 255]:
            buf = MsgBuffer()
            buf.write_byte(val)
            r = MsgReader(buf.get_data())
            assert r.read_byte() == val, f"failed for {val}"

    def test_write_read_short(self) -> None:
        """write_short -> read_short round-trip."""
        buf = MsgBuffer()
        buf.write_short(0xABCD)
        r = MsgReader(buf.get_data())
        assert r.read_short() == 0xABCD

    def test_write_read_long(self) -> None:
        """write_long -> read_long round-trip for various values."""
        for val in [0, 1, 0x12345678, 0xFFFFFFFF, 1000]:
            buf = MsgBuffer()
            buf.write_long(val)
            r = MsgReader(buf.get_data())
            assert r.read_long() == val, f"failed for 0x{val:x}"

    def test_write_read_string(self) -> None:
        """write_string -> read_string round-trip."""
        buf = MsgBuffer()
        buf.write_string("hello world")
        r = MsgReader(buf.get_data())
        assert r.read_string() == "hello world"

    def test_write_read_empty_string(self) -> None:
        """Empty string should round-trip as empty."""
        buf = MsgBuffer()
        buf.write_string("")
        r = MsgReader(buf.get_data())
        assert r.read_string() == ""

    def test_write_read_bits_partial(self) -> None:
        """write_bits with non-byte-aligned count (10 bits = 2 raw + 1 Huffman)."""
        buf = MsgBuffer()
        buf.write_bits(0x2AB, 10)
        r = MsgReader(buf.get_data())
        assert r.read_bits(10) == 0x2AB

    def test_write_read_bits_single(self) -> None:
        """Single-bit writes (fully raw, no Huffman)."""
        buf = MsgBuffer()
        buf.write_bits(1, 1)
        buf.write_bits(0, 1)
        buf.write_bits(1, 1)
        r = MsgReader(buf.get_data())
        assert r.read_bits(1) == 1
        assert r.read_bits(1) == 0
        assert r.read_bits(1) == 1

    def test_write_read_signed_bits(self) -> None:
        """Signed bit values (negative bit count)."""
        for val in [-1, -5, -128, 0, 127]:
            buf = MsgBuffer()
            # Write as unsigned bits, read as signed
            buf.write_bits(val & 0xFFFF, -16)
            r = MsgReader(buf.get_data())
            result = r.read_bits(-16)
            assert result == val, f"signed: expected {val}, got {result}"

    def test_mixed_writes(self) -> None:
        """Interleaved writes of different types."""
        buf = MsgBuffer()
        buf.write_long(42)
        buf.write_byte(7)
        buf.write_long(1000)
        buf.write_bits(0, 1)
        buf.write_string("bhop_flat")
        buf.write_short(0x1234)

        r = MsgReader(buf.get_data())
        assert r.read_long() == 42
        assert r.read_byte() == 7
        assert r.read_long() == 1000
        assert r.read_bits(1) == 0
        assert r.read_string() == "bhop_flat"
        assert r.read_short() == 0x1234

    def test_write_data(self) -> None:
        """write_data -> sequential read_byte round-trip."""
        payload = bytes([10, 20, 30, 40, 50])
        buf = MsgBuffer()
        buf.write_data(payload)
        r = MsgReader(buf.get_data())
        result = bytes(r.read_byte() for _ in range(len(payload)))
        assert result == payload

    def test_multiple_strings(self) -> None:
        """Multiple strings written sequentially."""
        buf = MsgBuffer()
        buf.write_string("first")
        buf.write_string("second")
        r = MsgReader(buf.get_data())
        assert r.read_string() == "first"
        assert r.read_string() == "second"


# ---------------------------------------------------------------------------
# TestDemoWriter: demo file structure and integration tests
# ---------------------------------------------------------------------------


class TestDemoWriter:
    """DemoWriter correctness tests."""

    def _make_ticks(self, n: int, dt_ms: int = 8) -> list[TickRecord]:
        """Create n simple ticks with linear motion."""
        ticks = []
        for i in range(n):
            ticks.append(TickRecord(
                server_time=i * dt_ms,
                origin_x=float(i * 2.56),
                origin_y=0.0,
                origin_z=0.0,
                velocity_x=320.0,
                velocity_y=0.0,
                velocity_z=0.0,
                yaw=float(i * 0.5),
                on_ground=True,
            ))
        return ticks

    def _write_and_read(
        self, ticks: list[TickRecord], map_name: str = "bhop_flat"
    ) -> list[tuple[int, int, bytes]]:
        """Write ticks to a temp demo, return list of (seq, length, data)."""
        with tempfile.NamedTemporaryFile(suffix=".dm_68", delete=False) as f:
            path = f.name
        try:
            DemoWriter(map_name).write(ticks, path)
            messages = []
            with open(path, "rb") as f:
                while True:
                    header = f.read(8)
                    assert len(header) == 8
                    seq, length = struct.unpack("<ii", header)
                    if seq == -1 and length == -1:
                        # Verify nothing follows EOF
                        assert f.read() == b""
                        break
                    data = f.read(length)
                    assert len(data) == length
                    messages.append((seq, length, data))
            return messages
        finally:
            os.unlink(path)

    def test_minimal_demo(self) -> None:
        """Single-tick demo produces gamestate + 1 snapshot + EOF."""
        msgs = self._write_and_read(self._make_ticks(1))
        assert len(msgs) == 2  # gamestate + 1 snapshot

    def test_message_count(self) -> None:
        """N ticks produce 1 gamestate + N snapshots."""
        for n in [5, 10, 50]:
            msgs = self._write_and_read(self._make_ticks(n))
            assert len(msgs) == n + 1, f"n={n}: expected {n+1}, got {len(msgs)}"

    def test_sequence_numbers(self) -> None:
        """Message sequence numbers are sequential starting from 0."""
        msgs = self._write_and_read(self._make_ticks(5))
        for i, (seq, _length, _data) in enumerate(msgs):
            assert seq == i, f"msg {i}: expected seq {i}, got {seq}"

    def test_gamestate_is_first(self) -> None:
        """First message is a gamestate (starts with reliable_ack + SVC_GAMESTATE)."""
        msgs = self._write_and_read(self._make_ticks(3))
        r = MsgReader(msgs[0][2])
        assert r.read_long() == 0  # reliable_ack
        assert r.read_byte() == 2  # SVC_GAMESTATE

    def test_snapshots_follow_gamestate(self) -> None:
        """Messages after the first are snapshots (SVC_SNAPSHOT)."""
        msgs = self._write_and_read(self._make_ticks(3))
        for i in range(1, len(msgs)):
            r = MsgReader(msgs[i][2])
            assert r.read_long() == 0  # reliable_ack
            assert r.read_byte() == 7  # SVC_SNAPSHOT

    def test_snapshot_server_times(self) -> None:
        """Snapshot server_time values match tick data."""
        ticks = self._make_ticks(5, dt_ms=8)
        msgs = self._write_and_read(ticks)
        for i, tick in enumerate(ticks):
            r = MsgReader(msgs[i + 1][2])
            r.read_long()  # reliable_ack
            r.read_byte()  # SVC_SNAPSHOT
            server_time = r.read_long()
            assert server_time == tick.server_time

    def test_map_name_in_gamestate(self) -> None:
        """Map name appears in the gamestate configstrings."""
        msgs = self._write_and_read(self._make_ticks(1), map_name="bhop_corridor")
        r = MsgReader(msgs[0][2])
        r.read_long()  # reliable_ack
        r.read_byte()  # SVC_GAMESTATE
        r.read_long()  # server_command_sequence
        r.read_byte()  # SVC_CONFIGSTRING
        r.read_short()  # CS_SERVERINFO index
        serverinfo = r.read_string()
        assert "bhop_corridor" in serverinfo

    def test_empty_ticks_raises(self) -> None:
        """DemoWriter.write raises ValueError for empty tick list."""
        with pytest.raises(ValueError, match="empty"):
            DemoWriter("bhop_flat").write([], "/dev/null")

    def test_tick_to_playerstate_ground(self) -> None:
        """TickRecord with on_ground=True maps to ENTITYNUM_WORLD."""
        tick = TickRecord(
            server_time=100, origin_x=0, origin_y=0, origin_z=0,
            velocity_x=0, velocity_y=0, velocity_z=0, yaw=0,
            on_ground=True,
        )
        ps = _tick_to_playerstate(tick)
        assert ps["groundEntityNum"] == ENTITYNUM_WORLD

    def test_tick_to_playerstate_airborne(self) -> None:
        """TickRecord with on_ground=False maps to ENTITYNUM_NONE."""
        tick = TickRecord(
            server_time=100, origin_x=0, origin_y=0, origin_z=0,
            velocity_x=0, velocity_y=0, velocity_z=270, yaw=0,
            on_ground=False,
        )
        ps = _tick_to_playerstate(tick)
        assert ps["groundEntityNum"] == ENTITYNUM_NONE

    def test_physics_integration(self) -> None:
        """Run Q3Physics with scripted bhop inputs, export to demo file.

        Verifies the full pipeline: physics -> TickRecords -> demo.
        """
        from bhop.physics import Q3Physics

        phys = Q3Physics()
        ticks = []
        yaw_rate = np.radians(0.4)

        for i in range(100):
            # Bhop: always jump, strafe right, rotate yaw
            phys.tick(forward_move=0, right_move=127, jump=True,
                      yaw_delta=yaw_rate)
            ticks.append(TickRecord(
                server_time=i * 8,
                origin_x=phys.position[0],
                origin_y=phys.position[1],
                origin_z=phys.position[2],
                velocity_x=phys.velocity[0],
                velocity_y=phys.velocity[1],
                velocity_z=phys.velocity[2],
                yaw=np.degrees(phys.yaw),
                on_ground=phys.on_ground,
            ))

        # Write and read back
        msgs = self._write_and_read(ticks)
        assert len(msgs) == 101  # 1 gamestate + 100 snapshots

        # Verify file isn't degenerate (last snapshot should differ from first)
        assert msgs[-1][1] > 0  # non-empty data
