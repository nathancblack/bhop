# Q3 Demo File Format (.dm_68)

Reference documentation for the Quake III Arena demo format as implemented in `src/bhop/demo.py`. Derived from the ioquake3 source (`code/qcommon/msg.c`, `code/client/cl_main.c`, `code/game/bg_public.h`).

## Implementation Status

Fully implemented in `src/bhop/demo.py`:
- `HuffmanCodec`: static Huffman tree from pre-computed structure (matches Q3's msg_hData-initialized tree)
- `MsgBuffer` / `MsgReader`: MSG_WriteBits/ReadBits with Huffman encoding
- `write_delta_playerstate()`: full netfield delta encoding with float 0/13-bit/32-bit paths
- `_write_gamestate()`: configstrings (mapname, systeminfo, player)
- `_write_snapshot()`: playerstate delta + entity terminator
- `DemoWriter`: ties it all together into a framed .dm_68 file

Export scripts:
- `scripts/export_demo.py`: standalone pipeline (scripted bhop or trained model → .dm_68 + .map)
- `scripts/evaluate.py --export-demo`: export during model evaluation

## File Structure

A `.dm_68` file is a sequence of **messages** followed by an **EOF marker**.

```
[message 0]        ← gamestate
[message 1]        ← first snapshot
[message 2]        ← second snapshot
...
[message N]        ← last snapshot
[EOF marker]
```

### Message Framing

Each message:
```
int32_LE  sequence       server message sequence number
int32_LE  data_length    byte count of compressed data that follows
byte[]    data           Huffman-compressed message payload
```

### EOF Marker
```
int32_LE  -1             sentinel value
int32_LE  -1             sentinel value
```

Source: `CL_WriteDemoMessage()` in `cl_main.c`.

---

## Huffman Compression

All message payloads are compressed using Q3's Huffman coding, initialized from a static frequency table.

### Initialization

The codec is initialized from `msg_hData[256]` defined in `msg.c`. `MSG_initHuffman()` calls `Huff_addRef()` for each symbol `i` exactly `msg_hData[i]` times, building the initial tree via Vitter's adaptive algorithm.

### Key Properties

- **Static after initialization**: Despite being built with an adaptive algorithm, the tree is **NOT updated** during `MSG_WriteBits`/`MSG_ReadBits`. Q3's `MSG_WriteBits` calls `Huff_offsetTransmit` (encode only) without `Huff_addRef`. The tree is effectively a static Huffman code after `MSG_initHuffman()`. This was confirmed by reading Q3's `MSG_WriteBits` source -- no `Huff_addRef` call exists there.
- **Shared state**: Both encoder (compressor) and decoder (decompressor) use trees initialized identically from `msg_hData`.
- **Bit ordering**: LSB-first within each byte. Bit 0 of a byte is written/read first.

**Implementation note**: Because the tree is static, our implementation uses a pre-computed tree structure (from `jfedor2/quake3-proxy-aimbot`) rather than performing ~960K `Huff_addRef` calls. The tree is loaded once at module import and reused for all demos. Encoding uses a precomputed lookup table (symbol → code bits).

### MSG_WriteBits Behavior

Q3's `MSG_WriteBits(msg, value, bits)` has dual behavior:
- **Partial bits** (`bits % 8 != 0`): the remainder bits are written **raw** (uncompressed, one bit at a time)
- **Full bytes** (remaining after partial): each byte is **Huffman-encoded** through the codec

Example: `MSG_WriteBits(msg, value, 10)`:
1. Write 2 bits raw (value & 0x3)
2. Write 1 byte Huffman-encoded ((value >> 2) & 0xFF)

This means `MSG_WriteByte` (8 bits) is fully Huffman-encoded, `MSG_WriteLong` (32 bits) is 4 Huffman-encoded bytes, but `MSG_WriteBits(val, 1)` is one raw bit.

Source: `MSG_WriteBits()` in `msg.c`.

---

## Message Types

Each message payload (after Huffman decompression) contains:

```
int32   reliable_ack    reliable command acknowledge (0 for demos)
byte    svc_command     first command type
...     command data    command-specific fields
byte    svc_command     next command (optional)
...
byte    SVC_EOF (8)     end of message
```

### SVC Command Values

```
SVC_GAMESTATE     = 2    initial game state (first message)
SVC_CONFIGSTRING  = 3    configuration string (inside gamestate)
SVC_BASELINE      = 4    entity baseline (inside gamestate)
SVC_SNAPSHOT      = 7    world snapshot (subsequent messages)
SVC_EOF           = 8    end of message / end of gamestate loop
```

---

## Gamestate Message (svc_gamestate)

The first message in a demo. Establishes the game session.

```
byte    SVC_GAMESTATE (2)
int32   server_command_sequence

— repeated configstrings/baselines: —
  byte    SVC_CONFIGSTRING (3)
  int16   index                    configstring index
  string  value                    null-terminated string
  ... or ...
  byte    SVC_BASELINE (4)
  bits(10) entity_number           GENTITYNUM_BITS
  ...      entity delta data       MSG_ReadDeltaEntity from null baseline
— end repeat —

byte    SVC_EOF (8)               ends the gamestate loop
int32   client_num                 local player's client number
int32   checksum_feed              for pure server validation (0 for demos)
```

### Required Configstrings

| Index | Name | Example Value |
|-------|------|---------------|
| 0 | CS_SERVERINFO | `\mapname\bhop_corridor\g_gametype\0` |
| 1 | CS_SYSTEMINFO | `\sv_serverid\0` |
| 544+N | CS_PLAYERS + clientNum | `\n\Player` |

Source: `CL_ParseGamestate()` in `cl_main.c`.

---

## Snapshot Message (svc_snapshot)

Each subsequent message contains a world state snapshot.

```
byte    SVC_SNAPSHOT (7)
int32   server_time               milliseconds since server start
byte    delta_num                 delta reference (0 = not delta-compressed)
byte    snap_flags                snapshot flags (0 = normal)
byte    areamask_len              byte count of areamask
byte[]  areamask                  PVS area visibility mask
...     playerstate delta         MSG_WriteDeltaPlayerstate
bits(10) ENTITYNUM_NONE (1023)    entity list terminator
```

For the first snapshot, `delta_num = 0` means full state (no delta reference). Subsequent snapshots can delta from the previous.

Source: `CL_ParseSnapshot()` in `cl_main.c`.

---

## Playerstate Delta Encoding

Uses a **netfield table** (`playerStateFields[]`) defining field order and bit widths.

### Algorithm (MSG_WriteDeltaPlayerstate)

1. Compare `from_ps` and `to_ps` across all fields in order
2. Find `lc` = index of last changed field + 1
3. Write `lc` as byte (0 means nothing changed)
4. For each field index 0..lc-1:
   - Write 1 bit: field changed (1) or unchanged (0)
   - If changed, write value using the field's encoding

### Field Encoding by Bit Width

- **bits == 0** (float field):
  - Value is 0.0: write single `0` bit
  - Value is non-zero, integer, fits in 13 bits: write `1`, `0`, then 13-bit biased integer (value + FLOAT_INT_BIAS)
  - Otherwise: write `1`, `1`, then 32-bit IEEE 754 float (as raw int32 bits)
- **bits > 0** (unsigned integer): write `bits` bits of value
- **bits < 0** (signed integer): write `|bits|` bits of value

### Constants

```
FLOAT_INT_BITS  = 13
FLOAT_INT_BIAS  = 4096    (2^(FLOAT_INT_BITS - 1))
```

### Stat Arrays

After the netfield delta, four 16-element arrays are encoded:
```
stats[16], persistant[16], ammo[16], powerups[16]
```

Each array:
1. Write 16-bit bitmask (which elements changed)
2. For each changed element: write 16-bit value (or 32-bit for powerups)

For our minimal demo, all arrays are zero (write 0x0000 bitmask for each).

### Playerstate Netfield Table

The field order must match Q3's `playerStateFields[]` exactly (32 fields). Fields with bits=0 are floats.

```
commandTime       32
origin[0]          0 (float)
origin[1]          0 (float)
bobCycle           8
velocity[0]        0 (float)
velocity[1]        0 (float)
viewangles[1]      0 (float)    ← yaw
viewangles[0]      0 (float)    ← pitch
weaponTime       -16 (signed)
origin[2]          0 (float)
velocity[2]        0 (float)
legsTimer          8
pm_time          -16 (signed)
eventSequence     16
torsoAnim          8
movementDir        4
events[0]          8
legsAnim           8
events[1]          8
pm_flags          16
groundEntityNum   10
weaponstate        4
eFlags            16
externalEvent     10
clientNum          8
viewangles[2]      0 (float)    ← roll
pm_type            8
damageEvent        8
damageYaw          8
damagePitch        8
damageCount        8
generic1           8
```

Source: `playerStateFields[]` in `msg.c`.

---

## Coordinate System

Q3 uses Z-up: X = forward, Y = left, Z = up. This matches our physics engine's coordinate system exactly, so no transform is needed.

### Angle Encoding

```c
ANGLE2SHORT(degrees) = (int)(degrees * 65536.0 / 360.0) & 0xFFFF
```

Viewangles in the playerstate are stored as **floats in degrees** (not shorts). The ANGLE2SHORT conversion is used in some network fields but not in the playerstate delta encoding, where viewangles are encoded as regular float fields.

---

## Entity Constants

```
GENTITYNUM_BITS  = 10
MAX_GENTITIES    = 1024    (2^GENTITYNUM_BITS)
ENTITYNUM_NONE   = 1023    (MAX_GENTITIES - 1)
ENTITYNUM_WORLD  = 1022    (MAX_GENTITIES - 2)
```

---

## References

- [ioquake3 source](https://github.com/ioquake/ioq3) -- authoritative
- [id-Software/Quake-III-Arena](https://github.com/id-Software/Quake-III-Arena/blob/master/code/qcommon/msg.c) -- original source
- [jfedor2/quake3-proxy-aimbot](https://github.com/jfedor2/quake3-proxy-aimbot) -- Python reference implementation
- [Q3 demo specs (elho.net)](http://www.elho.net/games/q3/q3dspecs.htm)
- [Q3 network protocol (jfedor.org)](https://www.jfedor.org/quake3/)
