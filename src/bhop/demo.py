"""Q3 demo file writer (.dm_68 format).

Implements the adaptive Huffman codec and bit-level I/O matching Q3's msg.c.
The codec is initialized from Q3's msg_hData[256] frequency table via a
pre-computed tree structure. Q3's MSG_WriteBits/MSG_ReadBits use this
pre-initialized tree WITHOUT per-symbol adaptive updates.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# msg_hData[256] -- symbol frequency table from Q3's msg.c
# Used to seed the Huffman tree via Huff_addRef in MSG_initHuffman().
# ---------------------------------------------------------------------------

MSG_HDATA: list[int] = [
    250315, 41193,  6292,  7106,  3730,  3750,  6110, 23283,  # 0-7
     33317,  6950,  7838,  9714,  9257, 17259,  3949,  1778,  # 8-15
      8288,  1604,  1590,  1663,  1100,  1213,  1238,  1134,  # 16-23
      1749,  1059,  1246,  1149,  1273,  4486,  2805,  3472,  # 24-31
     21819,  1159,  1670,  1066,  1043,  1012,  1053,  1070,  # 32-39
      1726,   888,  1180,   850,   960,   780,  1752,  3296,  # 40-47
     10630,  4514,  5881,  2685,  4650,  3837,  2093,  1867,  # 48-55
      2584,  1949,  1972,   940,  1134,  1788,  1670,  1206,  # 56-63
      5719,  6128,  7222,  6654,  3710,  3795,  1492,  1524,  # 64-71
      2215,  1140,  1355,   971,  2180,  1248,  1328,  1195,  # 72-79
      1770,  1078,  1264,  1266,  1168,   965,  1155,  1186,  # 80-87
      1347,  1228,  1529,  1600,  2617,  2048,  2546,  3275,  # 88-95
      2410,  3585,  2504,  2800,  2675,  6146,  3663,  2840,  # 96-103
     14253,  3164,  2221,  1687,  3208,  2739,  3512,  4796,  # 104-111
      4091,  3515,  5288,  4016,  7937,  6031,  5360,  3924,  # 112-119
      4892,  3743,  4566,  4807,  5852,  6400,  6225,  8291,  # 120-127
     23243,  7838,  7073,  8935,  5437,  4483,  3641,  5256,  # 128-135
      5312,  5328,  5370,  3492,  2458,  1694,  1821,  2121,  # 136-143
      1916,  1149,  1516,  1367,  1236,  1029,  1258,  1104,  # 144-151
      1245,  1006,  1149,  1025,  1241,   952,  1287,   997,  # 152-159
      1713,  1009,  1187,   879,  1099,   929,  1078,   951,  # 160-167
      1656,   930,  1153,  1030,  1262,  1062,  1214,  1060,  # 168-175
      1621,   930,  1106,   912,  1034,   892,  1158,   990,  # 176-183
      1175,   850,  1121,   903,  1087,   920,  1144,  1056,  # 184-191
      3462,  2240,  4397, 12136,  7758,  1345,  1307,  3278,  # 192-199
      1950,   886,  1023,  1112,  1077,  1042,  1061,  1071,  # 200-207
      1484,  1001,  1096,   915,  1052,   995,  1070,   876,  # 208-215
      1111,   851,  1059,   805,  1112,   923,  1103,   817,  # 216-223
      1899,  1872,   976,   841,  1127,   956,  1159,   950,  # 224-231
      7791,   954,  1289,   933,  1127,  3207,  1020,   927,  # 232-239
      1355,   768,  1040,   745,   952,   805,  1073,   740,  # 240-247
      1013,   805,  1008,   796,   996,  1057, 11457, 13504,  # 248-255
]

# ---------------------------------------------------------------------------
# Pre-computed Huffman tree structure
# From jfedor2/quake3-proxy-aimbot -- matches Q3's msg_hData-initialized tree.
#
# Format: 512 entries. Negative/zero = leaf (symbol = -value).
# Positive = internal node (left child index = value, right = value - 1).
# NYT (Not Yet Transmitted) node is at implicit index 512.
# ---------------------------------------------------------------------------

_SAVED_TREE: list[int] = [
    2, 4, 6, 8, 10, 0, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36,
    38, 40, 42, 44, 46, 48, 50, 52, 54, 56, -1, 58, 60, 62, 64, 66, -8, 68,
    70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, -7, -128, 94, 96, -32,
    98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, -13, 120, 122, 124,
    126, 128, 130, 132, 134, 136, 138, 140, 142, -104, 144, 146, 148, 150,
    -255, 152, 154, 156, 158, 160, 162, 164, -195, 166, 168, -254, 170, 172,
    174, 176, 178, -48, 180, 182, 184, 186, 188, -11, 190, 192, 194, -12,
    196, 198, 200, 202, -131, 204, 206, 208, 210, 212, -127, -16, 214, 216,
    218, 220, -116, 222, -10, -129, -232, -196, 224, 226, 228, 230, 232, 234,
    236, 238, -66, 240, -3, 242, -130, 244, 246, -9, 248, 250, 252, -67, 254,
    256, 258, 260, -125, 262, -2, -126, 264, -101, -65, -6, -117, 266, -50,
    -124, -64, 268, 270, -132, 272, -138, 274, -118, -137, -136, 276, -114,
    -135, 278, 280, 282, 284, 286, 288, -120, 290, -123, -111, 292, 294, -52,
    296, 298, -122, 300, -49, -29, -133, 302, 304, 306, 308, -194, 310, 312,
    314, 316, 318, 320, 322, 324, -112, 326, 328, -115, 330, -14, 332, -119,
    334, 336, -53, 338, 340, -69, 342, -5, -121, 344, -4, -68, 346, -102,
    348, -134, 350, -97, 352, 354, -113, -110, 356, -139, -31, -192, 358,
    360, 362, 364, 366, -47, -199, 368, -95, 370, -108, -237, 372, -105, 374,
    376, 378, 380, -103, -30, -99, -109, 382, 384, -51, -100, 386, -92, 388,
    -56, 390, -94, 392, 394, -98, 396, 398, 400, -140, 402, 404, -96, 406,
    408, 410, 412, 414, 416, 418, 420, 422, 424, -193, 426, 428, -106, -72,
    430, 432, 434, -76, 436, 438, 440, 442, 444, 446, -143, 448, 450, 452,
    -54, 454, 456, 458, 460, -93, 462, 464, 466, 468, 470, -58, 472, -200,
    -57, 474, 476, -144, 478, 480, -224, 482, -225, -55, 484, 486, 488, 490,
    -142, 492, -61, 494, -15, -80, 496, -46, -24, 498, -40, -160, 500, -141,
    -107, -62, -34, -19, 502, -168, -176, 504, -17, 506, -91, -18, 508, -90,
    -71, -146, -70, 510, -208, -147, -240, -74, -88, -197, -78, -198, -234,
    -158, -28, -83, -82, -172, -150, -77, -26, -152, -156, -22, -148, -89,
    -174, -21, -63, -79, -162, -87, -42, -184, -84, -230, -33, -182, -86,
    -170, -154, -27, -145, -190, -73, -23, -60, -236, -228, -186, -203,
    -220, -216, -178, -151, -222, -20, -164, -210, -188, -81, -166, -204,
    -246, -207, -39, -214, -35, -173, -206, -175, -218, -25, -253, -191,
    -38, -212, -36, -205, -242, -180, -171, -149, -155, -202, -238, -248,
    -37, -161, -250, -153, -209, -159, -252, -213, -183, -226, -75, -85,
    -44, -229, -233, -157, -244, -167, -231, -59, -235, -169, -177, -165,
    -239, -221, -189, -211, -179, -187, -181, -41, -201, -163, -215, -217,
    -185, -43, -227, -223, -249, -219, -245, -251, -45, -241, -243, 512,
    -247,
]


# ---------------------------------------------------------------------------
# Huffman Codec
# ---------------------------------------------------------------------------

class _HuffNode:
    """Minimal tree node for static Huffman encoding/decoding."""
    __slots__ = ("symbol", "left", "right", "parent")

    def __init__(self) -> None:
        self.symbol: int | None = None  # None = internal node
        self.left: _HuffNode | None = None
        self.right: _HuffNode | None = None
        self.parent: _HuffNode | None = None


class HuffmanCodec:
    """Static Huffman codec matching Q3's msgHuff after MSG_initHuffman().

    Q3 seeds the Huffman tree from msg_hData[256] via repeated Huff_addRef
    calls, then uses the resulting tree WITHOUT further adaptive updates in
    MSG_WriteBits / MSG_ReadBits. This codec reconstructs that tree from a
    pre-computed structure (verified against jfedor2/quake3-proxy-aimbot).

    Bit ordering is LSB-first within each byte, matching Q3's add_bit/get_bit.
    """

    def __init__(self) -> None:
        n = len(_SAVED_TREE)  # 512
        nyt_idx = n  # 512 -- NYT's implicit index

        # Allocate nodes: 512 tree nodes + 1 NYT
        nodes = [_HuffNode() for _ in range(n + 1)]
        nodes[nyt_idx].symbol = 256  # NYT sentinel

        # Build tree from saved structure
        for i in range(n):
            v = _SAVED_TREE[i]
            if v <= 0:
                # Leaf node: symbol = -v
                nodes[i].symbol = -v
            else:
                # Internal node: left = v, right = v-1
                left = nodes[v]
                right = nodes[v - 1]
                nodes[i].left = left
                nodes[i].right = right
                left.parent = nodes[i]
                right.parent = nodes[i]

        self._root: _HuffNode = nodes[0]

        # Precompute encoding table: symbol -> (code_bits, num_bits)
        # code_bits is packed LSB-first: bit 0 is the root-side decision.
        self._codes: list[tuple[int, int]] = [(0, 0)] * 256
        for i in range(n + 1):
            node = nodes[i]
            if node.symbol is not None and node.symbol < 256:
                # Traverse leaf to root, collecting bits
                bits: list[int] = []
                cur = node
                while cur.parent is not None:
                    bits.append(1 if cur.parent.right is cur else 0)
                    cur = cur.parent
                # Reverse to root-to-leaf order, pack LSB-first
                bits.reverse()
                code = 0
                for j, b in enumerate(bits):
                    code |= b << j
                self._codes[node.symbol] = (code, len(bits))

    def encode_symbol(self, ch: int, fout: bytearray, offset: int) -> int:
        """Encode one byte through Huffman. Returns the new bit offset.

        Mirrors Q3's Huff_offsetTransmit + add_bit (LSB-first bit packing).
        """
        code, nbits = self._codes[ch]
        for i in range(nbits):
            byte_idx = offset >> 3
            bit_pos = offset & 7
            if bit_pos == 0:
                # Starting a new byte -- zero it first (matches Q3's add_bit)
                while len(fout) <= byte_idx:
                    fout.append(0)
                fout[byte_idx] = 0
            elif byte_idx >= len(fout):
                fout.append(0)
            fout[byte_idx] |= ((code >> i) & 1) << bit_pos
            offset += 1
        return offset

    def decode_symbol(
        self, fin: bytes | bytearray, offset: int
    ) -> tuple[int, int]:
        """Decode one byte from Huffman stream. Returns (symbol, new offset).

        Mirrors Q3's Huff_offsetReceive + get_bit (LSB-first bit reading).
        """
        node = self._root
        while node.symbol is None:
            bit = (fin[offset >> 3] >> (offset & 7)) & 1
            offset += 1
            node = node.right if bit else node.left  # type: ignore[assignment]
        return node.symbol, offset


def _make_q3_codec() -> HuffmanCodec:
    """Create the standard Q3 Huffman codec (cached at module level)."""
    return HuffmanCodec()


# Module-level codec instance -- reused across demo sessions since the tree
# is static (no adaptive updates in MSG_WriteBits/ReadBits).
_q3_codec = _make_q3_codec()


# ---------------------------------------------------------------------------
# Message Buffer (mirrors Q3's msg_t + MSG_Write* functions)
# ---------------------------------------------------------------------------

class MsgBuffer:
    """Q3-compatible message buffer with Huffman-compressed writes.

    Implements MSG_WriteBits and the MSG_Write{Byte,Short,Long,String,Data}
    convenience functions from Q3's msg.c.

    MSG_WriteBits behavior:
    - Partial bits (count % 8 != 0): written RAW, one bit at a time, LSB-first
    - Full bytes (remaining after partial): each byte Huffman-encoded
    - Negative bit counts indicate signed values (absolute value used for width)
    """

    def __init__(self, codec: HuffmanCodec | None = None) -> None:
        self._data = bytearray()
        self._bit = 0  # current bit position
        self._codec = codec or _q3_codec

    def write_bits(self, value: int, bits: int) -> None:
        """Write value using the specified number of bits (Q3's MSG_WriteBits).

        Negative bits = signed value (width is abs(bits)).
        Partial bits are written raw; full bytes go through Huffman.
        """
        if bits < 0:
            bits = -bits

        # Mask to bit width
        if bits < 32:
            value &= (1 << bits) - 1
        else:
            value &= 0xFFFFFFFF

        # Partial bits: write raw, LSB-first
        nbits = bits & 7
        for _ in range(nbits):
            self._put_bit(value & 1)
            value >>= 1
        bits -= nbits

        # Full bytes: Huffman-encode each one
        while bits > 0:
            self._bit = self._codec.encode_symbol(
                value & 0xFF, self._data, self._bit
            )
            value >>= 8
            bits -= 8

    def write_byte(self, value: int) -> None:
        """Write an unsigned byte (8 bits, Huffman-encoded)."""
        self.write_bits(value, 8)

    def write_short(self, value: int) -> None:
        """Write a 16-bit value (2 Huffman-encoded bytes)."""
        self.write_bits(value, 16)

    def write_long(self, value: int) -> None:
        """Write a 32-bit value (4 Huffman-encoded bytes)."""
        self.write_bits(value, 32)

    def write_string(self, s: str) -> None:
        """Write a null-terminated string (each char + NUL as Huffman bytes)."""
        for ch in s:
            c = ord(ch)
            if c == 0:
                break
            # Q3 clamps to 127 and replaces '%' with '.' for security
            if c > 127:
                c = ord(".")
            self.write_byte(c)
        self.write_byte(0)

    def write_data(self, data: bytes | bytearray) -> None:
        """Write raw bytes (each byte Huffman-encoded)."""
        for b in data:
            self.write_byte(b)

    def get_data(self) -> bytes:
        """Return the buffer contents as bytes.

        Must match Q3's cursize formula: (msg->bit >> 3) + 1.
        The reader sets readcount = (bit>>3)+1 after each read, and checks
        readcount > cursize for overflow. Using ceiling division would be
        one byte short when bit is a multiple of 8, triggering false overflow.
        """
        if self._bit == 0:
            return b""
        size = (self._bit >> 3) + 1
        while len(self._data) < size:
            self._data.append(0)
        return bytes(self._data[:size])

    @property
    def bit_position(self) -> int:
        """Current bit position in the buffer."""
        return self._bit

    def _put_bit(self, bit: int) -> None:
        """Write a single raw bit, LSB-first (Q3's Huff_putBit)."""
        byte_idx = self._bit >> 3
        bit_pos = self._bit & 7
        if bit_pos == 0:
            while len(self._data) <= byte_idx:
                self._data.append(0)
            self._data[byte_idx] = 0
        elif byte_idx >= len(self._data):
            self._data.append(0)
        self._data[byte_idx] |= bit << bit_pos
        self._bit += 1


class MsgReader:
    """Q3-compatible message reader with Huffman-compressed reads.

    Implements MSG_ReadBits and convenience functions for round-trip testing.
    """

    def __init__(
        self, data: bytes | bytearray, codec: HuffmanCodec | None = None
    ) -> None:
        self._data = data
        self._bit = 0
        self._codec = codec or _q3_codec

    def read_bits(self, bits: int) -> int:
        """Read a value using the specified number of bits (Q3's MSG_ReadBits).

        Negative bits = signed value.
        """
        signed = bits < 0
        if signed:
            bits = -bits

        value = 0

        # Partial bits: read raw
        nbits = bits & 7
        for i in range(nbits):
            value |= self._get_bit() << i
        bits -= nbits

        # Full bytes: Huffman-decode
        i = nbits
        while bits > 0:
            sym, self._bit = self._codec.decode_symbol(self._data, self._bit)
            value |= sym << i
            i += 8
            bits -= 8

        # Sign extend if needed
        if signed:
            total_bits = i
            if value & (1 << (total_bits - 1)):
                value -= 1 << total_bits

        return value

    def read_byte(self) -> int:
        return self.read_bits(8)

    def read_short(self) -> int:
        return self.read_bits(16)

    def read_long(self) -> int:
        return self.read_bits(32)

    def read_string(self) -> str:
        chars = []
        while True:
            c = self.read_byte()
            if c == 0:
                break
            chars.append(chr(c))
        return "".join(chars)

    def _get_bit(self) -> int:
        """Read a single raw bit (Q3's Huff_getBit)."""
        bit = (self._data[self._bit >> 3] >> (self._bit & 7)) & 1
        self._bit += 1
        return bit


# ---------------------------------------------------------------------------
# Q3 Demo Constants (from bg_public.h, msg.c, cl_main.c)
# ---------------------------------------------------------------------------

# SVC command types
SVC_GAMESTATE = 2
SVC_CONFIGSTRING = 3
SVC_BASELINE = 4
SVC_SNAPSHOT = 7
SVC_EOF = 8

# Entity constants
GENTITYNUM_BITS = 10
MAX_GENTITIES = 1 << GENTITYNUM_BITS  # 1024
ENTITYNUM_NONE = MAX_GENTITIES - 1    # 1023
ENTITYNUM_WORLD = MAX_GENTITIES - 2   # 1022

# Float encoding constants for playerstate delta
FLOAT_INT_BITS = 13
FLOAT_INT_BIAS = 1 << (FLOAT_INT_BITS - 1)  # 4096

# Configstring indices
CS_SERVERINFO = 0
CS_SYSTEMINFO = 1
CS_PLAYERS = 544


# ---------------------------------------------------------------------------
# Playerstate Netfield Table (from msg.c: playerStateFields[])
#
# Each entry is (field_name, bits). bits=0 means float field with special
# encoding (0/13-bit/32-bit). Negative bits = signed integer.
# The field ORDER is critical -- must match Q3's exactly.
# ---------------------------------------------------------------------------

PLAYERSTATE_FIELDS: list[tuple[str, int]] = [
    ("commandTime",     32),
    ("origin[0]",        0),  # float
    ("origin[1]",        0),  # float
    ("bobCycle",         8),
    ("velocity[0]",      0),  # float
    ("velocity[1]",      0),  # float
    ("viewangles[1]",    0),  # float (yaw)
    ("viewangles[0]",    0),  # float (pitch)
    ("weaponTime",     -16),  # signed
    ("origin[2]",        0),  # float
    ("velocity[2]",      0),  # float
    ("legsTimer",        8),
    ("pm_time",        -16),  # signed
    ("eventSequence",   16),
    ("torsoAnim",        8),
    ("movementDir",      4),
    ("events[0]",        8),
    ("legsAnim",         8),
    ("events[1]",        8),
    ("pm_flags",        16),
    ("groundEntityNum", 10),
    ("weaponstate",      4),
    ("eFlags",          16),
    ("externalEvent",   10),
    ("clientNum",        8),
    ("viewangles[2]",    0),  # float (roll)
    ("pm_type",          8),
    ("damageEvent",      8),
    ("damageYaw",        8),
    ("damagePitch",      8),
    ("damageCount",      8),
    ("generic1",         8),
]


def angle_to_short(degrees: float) -> int:
    """Convert degrees to Q3's 16-bit angle encoding (ANGLE2SHORT)."""
    return int(degrees * 65536.0 / 360.0) & 0xFFFF


def _float_to_int_bits(f: float) -> int:
    """Convert a Python float to its IEEE 754 32-bit representation as an int."""
    return struct.unpack("<I", struct.pack("<f", f))[0]


# ---------------------------------------------------------------------------
# TickRecord -- per-tick snapshot data for demo export
# ---------------------------------------------------------------------------

@dataclass
class TickRecord:
    """Per-tick physics state for demo export.

    Captures everything needed to write a playerstate delta in a snapshot.
    Coordinates are in Q3 native units (X forward, Y left, Z up).
    """

    server_time: int          # milliseconds since server start
    origin_x: float           # position
    origin_y: float
    origin_z: float
    velocity_x: float         # velocity
    velocity_y: float
    velocity_z: float
    yaw: float                # viewangles in degrees
    pitch: float = 0.0
    roll: float = 0.0
    on_ground: bool = True
    event_sequence: int = 0   # incremented on events (unused for now)


# ---------------------------------------------------------------------------
# Playerstate helpers
# ---------------------------------------------------------------------------

# Type alias: playerstate is a dict mapping field names to numeric values.
# Float fields store Python floats; integer fields store Python ints.
Playerstate = dict[str, float | int]


def _null_playerstate() -> Playerstate:
    """Return a zeroed playerstate (all fields 0 / 0.0)."""
    ps: Playerstate = {}
    for name, bits in PLAYERSTATE_FIELDS:
        ps[name] = 0.0 if bits == 0 else 0
    return ps


def _tick_to_playerstate(tick: TickRecord, client_num: int = 0) -> Playerstate:
    """Convert a TickRecord into a Q3 playerstate dict.

    Adds Z_OFFSET to origin_z because Q3's player bbox extends 24 units below
    the origin point. Our physics uses z=0 as ground, so origin_z=0 would put
    feet at z=-24 (inside the floor). The offset raises the player above the
    floor surface.
    """
    Z_OFFSET = 40  # units above physics z=0 (floor surface)
    ps = _null_playerstate()
    ps["commandTime"] = tick.server_time
    ps["origin[0]"] = tick.origin_x
    ps["origin[1]"] = tick.origin_y
    ps["origin[2]"] = tick.origin_z + Z_OFFSET
    ps["velocity[0]"] = tick.velocity_x
    ps["velocity[1]"] = tick.velocity_y
    ps["velocity[2]"] = tick.velocity_z
    ps["viewangles[0]"] = tick.pitch
    ps["viewangles[1]"] = tick.yaw
    ps["viewangles[2]"] = tick.roll
    ps["groundEntityNum"] = (
        ENTITYNUM_WORLD if tick.on_ground else ENTITYNUM_NONE
    )
    ps["clientNum"] = client_num
    ps["eventSequence"] = tick.event_sequence
    return ps


# ---------------------------------------------------------------------------
# Playerstate Delta Encoding (MSG_WriteDeltaPlayerstate from msg.c)
# ---------------------------------------------------------------------------

def _write_delta_field(buf: MsgBuffer, bits: int, value: float | int) -> None:
    """Write a single changed playerstate field value.

    Encoding depends on the field's bit width:
    - bits == 0: float field (0/13-bit/32-bit encoding)
    - bits > 0:  unsigned integer
    - bits < 0:  signed integer (written as |bits| unsigned bits)
    """
    if bits == 0:
        # Float field: 2-way encoding matching Q3's MSG_WriteDeltaPlayerstate.
        # Q3 write: bit 0 + 13-bit biased int, OR bit 1 + 32-bit raw float.
        # Q3 read:  bit 0 → read 13 bits - bias, bit 1 → read 32-bit float.
        # Zero is encoded as 13-bit biased integer (0 + FLOAT_INT_BIAS = 4096).
        fval = float(value)
        int_val = int(fval)
        if (
            fval == float(int_val)
            and 0 <= int_val + FLOAT_INT_BIAS < (1 << FLOAT_INT_BITS)
        ):
            # Small integer (includes zero): bit 0, then 13-bit biased value
            buf.write_bits(0, 1)
            buf.write_bits(int_val + FLOAT_INT_BIAS, FLOAT_INT_BITS)
        else:
            # Full 32-bit float: bit 1, then IEEE 754 bits
            buf.write_bits(1, 1)
            buf.write_bits(_float_to_int_bits(fval), 32)
    else:
        # Integer field
        buf.write_bits(int(value), bits)


def write_delta_playerstate(
    buf: MsgBuffer,
    from_ps: Playerstate | None,
    to_ps: Playerstate,
) -> None:
    """Write a playerstate delta (Q3's MSG_WriteDeltaPlayerstate).

    If from_ps is None, deltas from a zeroed playerstate.

    Algorithm:
    1. Find lc = last changed field index + 1
    2. Write lc as byte
    3. For each field 0..lc-1: write changed bit, then value if changed
    4. Write four 16-element stat arrays (all zero bitmasks for our demos)
    """
    if from_ps is None:
        from_ps = _null_playerstate()

    # Find last changed field index
    lc = 0
    for i, (name, _bits) in enumerate(PLAYERSTATE_FIELDS):
        if to_ps.get(name, 0) != from_ps.get(name, 0):
            lc = i + 1

    buf.write_byte(lc)

    if lc == 0:
        # Nothing changed -- still need to write stat arrays
        _write_stat_arrays(buf)
        return

    # Write field deltas
    for i in range(lc):
        name, bits = PLAYERSTATE_FIELDS[i]
        to_val = to_ps.get(name, 0)
        from_val = from_ps.get(name, 0)

        if to_val != from_val:
            buf.write_bits(1, 1)  # changed
            _write_delta_field(buf, bits, to_val)
        else:
            buf.write_bits(0, 1)  # unchanged

    # Stat arrays (all zeros for our minimal demos)
    _write_stat_arrays(buf)


def _write_stat_arrays(buf: MsgBuffer, health: int = 100) -> None:
    """Write the four 16-element stat arrays after playerstate fields.

    Q3's format: first a single bit indicating if ANY array changed.
    If 0, return immediately. If 1, then for each of the 4 arrays:
      - 1 bit: changed?
      - if changed: bitmask (MAX_STATS bits) + 16-bit values for set bits
    """
    if health <= 0:
        buf.write_bits(0, 1)  # no stat arrays changed
        return

    buf.write_bits(1, 1)  # arrays changed

    # stats: set STAT_HEALTH (0), STAT_ARMOR (3), STAT_WEAPONS (4)
    # STAT_WEAPONS is a bitmask: bit 1 = gauntlet (WP_GAUNTLET)
    buf.write_bits(1, 1)  # stats changed
    stat_bitmask = (1 << 0) | (1 << 3) | (1 << 4)  # health, armor, weapons
    buf.write_bits(stat_bitmask, 16)
    buf.write_short(health)   # STAT_HEALTH = 100
    buf.write_short(0)        # STAT_ARMOR = 0
    buf.write_short(1 << 1)   # STAT_WEAPONS = gauntlet bit

    # persistant: set PERS_SCORE(0)=0, PERS_SPAWN_COUNT(3)=1, PERS_TEAM(0 in Q3 is TEAM_FREE)
    # PERS_SPAWN_COUNT must be non-zero or cgame won't trigger initial spawn
    buf.write_bits(1, 1)  # persistant changed
    pers_bitmask = (1 << 3)  # PERS_SPAWN_COUNT (index 3)
    buf.write_bits(pers_bitmask, 16)
    buf.write_short(1)  # PERS_SPAWN_COUNT = 1

    buf.write_bits(0, 1)  # ammo: no change
    buf.write_bits(0, 1)  # powerups: no change


# ---------------------------------------------------------------------------
# Gamestate Message (first message in a demo)
# ---------------------------------------------------------------------------

def _write_gamestate(
    buf: MsgBuffer,
    map_name: str,
    client_num: int = 0,
) -> None:
    """Write a gamestate message payload (Q3's CL_ParseGamestate format).

    Structure:
      int32   reliable_ack (0)
      byte    SVC_GAMESTATE
      int32   server_command_sequence (0)
      [configstrings...]
      byte    SVC_EOF
      int32   client_num
      int32   checksum_feed (0)
    """
    buf.write_long(0)  # reliable_ack
    buf.write_byte(SVC_GAMESTATE)
    buf.write_long(0)  # server_command_sequence

    # CS_SERVERINFO (index 0): mapname and gametype
    buf.write_byte(SVC_CONFIGSTRING)
    buf.write_short(CS_SERVERINFO)
    buf.write_string(f"\\mapname\\{map_name}\\g_gametype\\0\\version\\baseq3-1")

    # CS_SYSTEMINFO (index 1): minimal server info
    buf.write_byte(SVC_CONFIGSTRING)
    buf.write_short(CS_SYSTEMINFO)
    buf.write_string("\\sv_serverid\\1234\\sv_pure\\0\\sv_maxclients\\8")

    # CS_GAME_VERSION (index 20): must match cgame's GAME_VERSION
    buf.write_byte(SVC_CONFIGSTRING)
    buf.write_short(20)  # CS_GAME_VERSION
    buf.write_string("baseq3-1")

    # CS_PLAYERS + clientNum: player info (name, model, head model, colors)
    # Must include 'model' key or cgame can't load player model.
    # 'sarge' is the default Q3 model available in baseq3 pak files.
    buf.write_byte(SVC_CONFIGSTRING)
    buf.write_short(CS_PLAYERS + client_num)
    buf.write_string(
        "\\n\\Player\\model\\sarge\\hmodel\\sarge"
        "\\c1\\4\\c2\\5\\hc\\100\\w\\0\\l\\0\\tt\\0\\tl\\0"
    )

    # End configstrings/baselines loop
    buf.write_byte(SVC_EOF)

    # Client number and checksum
    buf.write_long(client_num)
    buf.write_long(0)  # checksum_feed

    # End of server message (outer CL_ParseServerMessage loop)
    buf.write_byte(SVC_EOF)


# ---------------------------------------------------------------------------
# Snapshot Message (subsequent messages in a demo)
# ---------------------------------------------------------------------------

def _write_snapshot(
    buf: MsgBuffer,
    server_time: int,
    from_ps: Playerstate | None,
    to_ps: Playerstate,
    delta_num: int = 0,
) -> None:
    """Write a snapshot message payload (Q3's CL_ParseSnapshot format).

    Structure:
      int32   reliable_ack (0)
      byte    SVC_SNAPSHOT
      int32   server_time
      byte    delta_num (0 = full, >0 = delta from that snapshot)
      byte    snap_flags (0)
      byte    areamask_len
      byte[]  areamask
      ...     playerstate delta
      bits(10) ENTITYNUM_NONE  (entity list terminator)
    """
    buf.write_long(0)  # reliable_ack
    buf.write_byte(SVC_SNAPSHOT)
    buf.write_long(server_time)
    buf.write_byte(delta_num)
    buf.write_byte(0)  # snap_flags

    # Areamask: 1 byte, value 0x00. Q3's areamask is a BLOCK mask (inverted):
    # set bit = area NOT visible. 0x00 = all areas visible.
    buf.write_byte(1)   # areamask_len
    buf.write_byte(0)   # areamask data -- 0 = don't block any areas

    # Playerstate delta
    write_delta_playerstate(buf, from_ps, to_ps)

    # Entity list terminator: ENTITYNUM_NONE (1023) as 10 bits
    buf.write_bits(ENTITYNUM_NONE, GENTITYNUM_BITS)

    # End of message
    buf.write_byte(SVC_EOF)


# ---------------------------------------------------------------------------
# DemoWriter -- ties everything together into a .dm_68 file
# ---------------------------------------------------------------------------

class DemoWriter:
    """Writes a Q3 .dm_68 demo file from a sequence of TickRecords.

    Usage:
        writer = DemoWriter("bhop_flat")
        writer.write(ticks, "output.dm_68")

    File structure:
        [message 0]  gamestate (configstrings, map name, client)
        [message 1]  first snapshot (full, delta_num=0)
        [message 2+] subsequent snapshots (delta from previous)
        [EOF marker] int32(-1) + int32(-1)

    Each message is framed as:
        int32_LE(sequence) + int32_LE(data_length) + byte[data]
    """

    def __init__(self, map_name: str, client_num: int = 0) -> None:
        self._map_name = map_name
        self._client_num = client_num

    def write(self, ticks: list[TickRecord], path: str) -> None:
        """Write a complete .dm_68 demo file."""
        if not ticks:
            raise ValueError("ticks must not be empty")

        with open(path, "wb") as f:
            seq = 0

            # Message 0: gamestate
            buf = MsgBuffer()
            _write_gamestate(buf, self._map_name, self._client_num)
            self._write_message(f, seq, buf)
            seq += 1

            # Message 1: first snapshot (full, delta_num=0)
            prev_ps = None
            cur_ps = _tick_to_playerstate(ticks[0], self._client_num)
            buf = MsgBuffer()
            _write_snapshot(buf, ticks[0].server_time, prev_ps, cur_ps,
                            delta_num=0)
            self._write_message(f, seq, buf)
            seq += 1
            prev_ps = cur_ps

            # Messages 2+: non-delta snapshots (delta_num=0 for reliability)
            for tick in ticks[1:]:
                cur_ps = _tick_to_playerstate(tick, self._client_num)
                buf = MsgBuffer()
                _write_snapshot(buf, tick.server_time, None, cur_ps,
                                delta_num=0)
                self._write_message(f, seq, buf)
                seq += 1
                prev_ps = cur_ps

            # EOF marker
            f.write(struct.pack("<ii", -1, -1))

    @staticmethod
    def _write_message(f, sequence: int, buf: MsgBuffer) -> None:
        """Write a single framed message: seq(i32) + len(i32) + data."""
        data = buf.get_data()
        f.write(struct.pack("<ii", sequence, len(data)))
        f.write(data)
