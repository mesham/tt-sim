"""Minimal Blackhole bring-up: construct the device and move data over the NoC.

A ``driver/simple``-style standalone script (no tt-metal wire bridge — that is
Phase 5). It constructs a Blackhole device (one DRAM tile + one Tensix tile),
then issues a NoC read that copies a buffer from DRAM into the Tensix core's L1
and checks it arrived intact — exercising the Blackhole grid (17x12), the
Blackhole NoC address encoding, directory wiring and NoC-1 mirror end to end.

Blackhole holds the destination coordinate in a dedicated register
(``NOC_TARG_ADDR_HI`` / ``NOC_RET_ADDR_HI``) as ``(Y << 6) | X``, rather than
packing it into the MID address register the way Wormhole does.

Run:  python3 -m driver.blackhole.bringup
"""

from tt_sim.device.blackhole import Blackhole
from tt_sim.util.conversion import conv_to_bytes

# NUI command-register offsets within a tile's NoC 0 region (base 0xFFB20000).
NUI_BASE = 0xFFB20000
REG_TARGET_ADDR_LOW = 0x0
REG_TARGET_ADDR_MID = 0x4
REG_TARGET_ADDR_HI = 0x8  # Blackhole coordinate register
REG_RET_ADDR_LOW = 0xC
REG_RET_ADDR_MID = 0x10
REG_RET_ADDR_HI = 0x14  # Blackhole coordinate register
REG_PACKET_TAG = 0x18
REG_CTRL = 0x1C
REG_AT_LEN_BE = 0x20
REG_CMD_CTRL = 0x28


def _coord_reg(x, y):
    """Encode a tile coord for the Blackhole coordinate register: ``(Y<<6)|X``."""
    return (y << 6) | x


def noc_read(device, initiator_coord, src_coord, src_off, dst_off, size):
    """Issue a NoC read on ``initiator_coord``'s NoC 0: copy ``size`` bytes from
    ``src_coord``'s memory at ``src_off`` into this tile's L1 at ``dst_off``."""

    def reg(off, val):
        device.write(initiator_coord, NUI_BASE + off, conv_to_bytes(val), 4)

    reg(REG_TARGET_ADDR_LOW, src_off)
    reg(REG_TARGET_ADDR_MID, 0)  # address high bits only (0 for a low offset)
    reg(REG_TARGET_ADDR_HI, _coord_reg(*src_coord))
    reg(REG_RET_ADDR_LOW, dst_off)
    reg(REG_RET_ADDR_MID, 0)
    reg(REG_RET_ADDR_HI, _coord_reg(*initiator_coord))
    reg(REG_PACKET_TAG, 0)  # transaction id 0
    reg(REG_CTRL, 0)  # mode 0 = read
    reg(REG_AT_LEN_BE, size)
    reg(REG_CMD_CTRL, 1)  # trigger initiate()


def main():
    device = Blackhole()
    dram = device.dram_tiles[0].get_coord_pair()
    tensix = device.tensix_tiles[0].get_coord_pair()

    payload = bytes((i * 7) % 256 for i in range(64))
    device.write(dram, 0x100, payload, len(payload))

    noc_read(device, tensix, dram, src_off=0x100, dst_off=0x200, size=len(payload))
    device.run(30)  # pump the clock so the request and its response propagate

    got = device.read(tensix, 0x200, len(payload))
    device.shutdown()

    assert got == payload, (
        f"NoC read mismatch:\n  want {payload.hex()}\n  got  {got.hex()}"
    )
    print(
        f"Completed successfully: NoC-read {len(payload)} bytes "
        f"DRAM{dram} -> Tensix{tensix} L1 on Blackhole"
    )


if __name__ == "__main__":
    main()
