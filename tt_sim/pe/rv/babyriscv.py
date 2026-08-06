from enum import IntEnum

from tt_sim.memory.memory import MemorySpace
from tt_sim.pe.rv.cost import make_cost_state
from tt_sim.pe.rv.isa.a_isa import RV_ZAAMO_ISA
from tt_sim.pe.rv.isa.b_isa import RV_ZBA_ISA, RV_ZBB_ISA
from tt_sim.pe.rv.isa.guard_isa import RV_F_GUARD_ISA, RV_V_GUARD_ISA
from tt_sim.pe.rv.isa.zfh_isa import RV_ZFH_ISA
from tt_sim.pe.rv.rv32 import RV32IM_TT
from tt_sim.pe.rv.spin import (
    SPIN_IDLE,
    FirmwareSpin,
    firmware_idle_debug_from_env,
    firmware_idle_enabled_from_env,
)
from tt_sim.pe.tensix.util import TensixConfigurationConstants
from tt_sim.util.bits import get_nth_bit
from tt_sim.util.conversion import conv_to_uint32

# Maps arch-profile ISA-extension names to the ISA classes that implement them.
# Kept here (not in the arch profile) so the profile stays free of pe.rv imports.
# ``f_guard`` / ``v_guard`` are placeholders that raise NotImplementedError for
# the not-yet-modelled F single-precision and V extensions (see guard_isa.py);
# ``zfh`` is the real half-precision unit (see zfh_isa.py).
ISA_EXTENSION_REGISTRY = {
    "zba": RV_ZBA_ISA,
    "zbb": RV_ZBB_ISA,
    "zaamo": RV_ZAAMO_ISA,
    "zfh": RV_ZFH_ISA,
    "f_guard": RV_F_GUARD_ISA,
    "v_guard": RV_V_GUARD_ISA,
}

# Extensions that require the floating-point register file to be allocated.
_FP_EXTENSIONS = frozenset({"zfh", "f_guard"})

# RISCV_DEBUG_REG_SOFT_RESET_0, in each tile's tile-control region.
SOFT_RESET_ADDR = 0xFFB121B0


class BabyRISCVCoreType(IntEnum):
    BRISC = 0
    NCRISC = 1
    TRISC0 = 2
    TRISC1 = 3
    TRISC2 = 4
    ERISC = 5


class BabyRISCV(RV32IM_TT):
    #: True while the firmware-loop recogniser has this core parked in a
    #: verified pure poll loop (see ``tt_sim/pe/rv/spin.py``). Read by the
    #: tile-level ``next_wake_cycle`` fast reject, so it is a plain class
    #: attribute that costs nothing until a core actually parks.
    spin_parked = False

    # Reset bit position within ``RISCV_DEBUG_REG_SOFT_RESET_0`` (at offset
    # 0xFFB121B0 in each tile's tile-control region). Each tile has its own
    # register, so ERISC (in eth tiles) reuses bit 11 — the same constant
    # name ``RISCV_SOFT_RESET_0_BRISC`` per the WormholeB0 EthernetTile/
    # SoftReset.md docs — without colliding with BRISC in Tensix tiles.
    CORE_TYPE_TO_SOFT_RESET_BIT = {
        BabyRISCVCoreType.BRISC: 11,
        BabyRISCVCoreType.NCRISC: 18,
        BabyRISCVCoreType.TRISC0: 12,
        BabyRISCVCoreType.TRISC1: 13,
        BabyRISCVCoreType.TRISC2: 14,
        BabyRISCVCoreType.ERISC: 11,
    }

    def __init__(
        self,
        core_type,
        memory_spaces,
        snoop=False,
        reset_pc_debug_regs=False,
        start_address=None,
        isa_extensions=(),
        arch=None,
    ):
        # Blackhole moves the NCRISC/TRISC reset-PC override out of the Tensix
        # backend config (Wormhole) into the RISCV_DEBUG_REG tile-control block.
        self.reset_pc_debug_regs = reset_pc_debug_regs
        self.core_type = core_type
        self.soft_active = False
        self.soft_reset_mask = 1 << BabyRISCV.CORE_TYPE_TO_SOFT_RESET_BIT[core_type]
        # Resolved lazily by _read_soft_reset(): (component, offset).
        self._soft_reset_src = None
        if core_type == BabyRISCVCoreType.BRISC:
            core_id = 0
            start_addr = 0x0
        elif core_type == BabyRISCVCoreType.NCRISC:
            core_id = 4
            start_addr = 0x12000
        elif core_type == BabyRISCVCoreType.TRISC0:
            core_id = 1
            start_addr = 0x06000
        elif core_type == BabyRISCVCoreType.TRISC1:
            core_id = 2
            start_addr = 0x0A000
        elif core_type == BabyRISCVCoreType.TRISC2:
            core_id = 3
            start_addr = 0x0E000
        elif core_type == BabyRISCVCoreType.ERISC:
            # Eth tile boot vector lives at L1 offset 0 (the "Tenstorrent
            # code/data, read-only" 1 KiB region per the EthernetTile/L1.md
            # ISA docs). Mirrors BRISC's convention in Tensix tiles.
            core_id = 5
            start_addr = 0x0
        # Blackhole lays the baby-RISC firmware out at different L1 bases than
        # Wormhole (and, having no IRAM constraints, boots NCRISC from L1 rather
        # than 0x12000). When the arch profile supplies a firmware base, it is
        # the core's default boot PC (used unless a reset-PC override is set).
        if start_address is not None:
            start_addr = start_address
        extensions = [ISA_EXTENSION_REGISTRY[name] for name in isa_extensions]
        fp_registers = any(name in _FP_EXTENSIONS for name in isa_extensions)
        super().__init__(
            start_addr,
            memory_spaces,
            extensions,
            snoop=snoop,
            core_id=core_id,
            fp_registers=fp_registers,
        )
        self.core_label = core_type.name
        # The cycle-cost model for the load/store path. ``None`` unless the
        # caller names an architecture *and* ``TT_SIM_COST_MODEL`` is set, so
        # cores built outside a device (driver/simple, the ISA unit tests) are
        # never charged. See tt_sim/pe/rv/cost.py.
        self.rv_cost = make_cost_state(arch)
        # Firmware-loop recognition (``tt_sim/pe/rv/spin.py``): parks a core
        # spinning in a verified pure poll loop so its tile can go dormant.
        # ``None`` when killed via TT_SIM_FIRMWARE_IDLE=0.
        self._spin = (
            FirmwareSpin(debug=firmware_idle_debug_from_env())
            if firmware_idle_enabled_from_env()
            else None
        )
        # Bound once: the plain RV32I tick the spin state machine drives.
        self._base_tick = super().clock_tick

    def get_start_address(self):
        """
        Retrieves the start address of the Baby RISC-V core, this depends on which core it is.
        BRISC is always 0x0 which is the default, whereas the others have a default start address
        (already set) but this can be overridden by what is stored with the Tensix backend configuration
        https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/SoftReset.md
        """
        if self.core_type == BabyRISCVCoreType.BRISC:
            return self.start_address
        if self.core_type == BabyRISCVCoreType.ERISC:
            # Eth tiles have no Tensix backend config (no FPU/SFPU/packer),
            # so the reset-PC-override mechanism that lives at
            # 0xFFEF0000 in a Tensix tile simply doesn't exist here. ERisc
            # always boots from its compile-time start_address.
            return self.start_address

        if self.reset_pc_debug_regs:
            return self._blackhole_reset_pc()

        TENSIX_BACKEND_CONFIG_BASE = 0xFFEF_0000

        if self.core_type == BabyRISCVCoreType.NCRISC:
            override_key = "NCRISC_RESET_PC_OVERRIDE_Reset_PC_Override_en"
            enabled_bit = 0
        else:
            override_key = "TRISC_RESET_PC_OVERRIDE_Reset_PC_Override_en"
            enabled_bit = self.core_type - 2

        override_tensix_config_addr_offset = (
            TensixConfigurationConstants.get_addr32(override_key) * 4
        )
        override_flag = conv_to_uint32(
            self.visible_memory.read(
                TENSIX_BACKEND_CONFIG_BASE + override_tensix_config_addr_offset, 4
            )
        )
        override_flag = TensixConfigurationConstants.parse_raw_config_value(
            override_flag, override_key
        )
        override_enabled = get_nth_bit(override_flag, enabled_bit)

        if override_enabled:
            if self.core_type == BabyRISCVCoreType.NCRISC:
                pc_key = "NCRISC_RESET_PC_PC"
            elif self.core_type == BabyRISCVCoreType.TRISC0:
                pc_key = "TRISC_RESET_PC_SEC0_PC"
            elif self.core_type == BabyRISCVCoreType.TRISC1:
                pc_key = "TRISC_RESET_PC_SEC1_PC"
            elif self.core_type == BabyRISCVCoreType.TRISC2:
                pc_key = "TRISC_RESET_PC_SEC2_PC"
            else:
                raise Exception("Unknown core type")
            pc_val_addr_offset = TensixConfigurationConstants.get_addr32(pc_key) * 4
            raw_pc = conv_to_uint32(
                self.visible_memory.read(
                    TENSIX_BACKEND_CONFIG_BASE + pc_val_addr_offset, 4
                )
            )
            return TensixConfigurationConstants.parse_raw_config_value(raw_pc, pc_key)
        else:
            return self.start_address

    def _blackhole_reset_pc(self):
        """Blackhole NCRISC/TRISC reset-PC override, read from the tile-control
        RISCV_DEBUG_REG block (base 0xFFB12000) rather than the Tensix backend
        config. Per tt-metal blackhole/tensix.h: TRISC0/1/2_RESET_PC at
        0x228/0x22C/0x230, TRISC_RESET_PC_OVERRIDE (one enable bit per TRISC) at
        0x234; NCRISC_RESET_PC at 0x238, NCRISC_RESET_PC_OVERRIDE (bit 0) at
        0x23C. When the override bit is clear the core boots from its default.
        """
        DEBUG_BASE = 0xFFB1_2000

        def rd(offset):
            return conv_to_uint32(self.visible_memory.read(DEBUG_BASE + offset, 4))

        if self.core_type == BabyRISCVCoreType.NCRISC:
            if not get_nth_bit(rd(0x23C), 0):
                return self.start_address
            return rd(0x238)

        idx = self.core_type - BabyRISCVCoreType.TRISC0  # 0, 1, 2
        if not get_nth_bit(rd(0x234), idx):
            return self.start_address
        return rd(0x228 + idx * 4)

    def _read_soft_reset(self):
        """Value of ``RISCV_DEBUG_REG_SOFT_RESET_0``.

        Every core polls this on every cycle, running or not, so the memory-map
        traversal (interval lookup + polymorphic dispatch + event publication)
        is resolved to the owning component once and then read directly. The
        traversal was ~28% of the cost of ticking an otherwise idle device.
        A space with snoop addresses configured keeps the slow path, so
        ``add_snoop`` still sees the access.
        """
        src = self._soft_reset_src
        if src is None:
            memory = self.visible_memory
            addr_range, target = memory.memory_map.locate(SOFT_RESET_ADDR)
            if (
                target is None
                or isinstance(target, MemorySpace)
                or memory.snoop_addresses
            ):
                return conv_to_uint32(memory.read(SOFT_RESET_ADDR, 4))
            src = self._soft_reset_src = (target, SOFT_RESET_ADDR - addr_range.low)
        return int.from_bytes(src[0].read(src[1], 4), "little")

    def is_clock_idle(self):
        """Idle while held in soft reset with the reset already taken.

        ``soft_active`` False means the last tick either saw the core in reset
        or has not run one yet; the reset-register read then confirms it is
        still held. Both halves are needed: a core that is out of reset with
        ``soft_active`` False is about to boot, and a core in reset with
        ``soft_active`` True still owes one tick to clear the flag.
        """
        if self.soft_active:
            return False
        return self._read_soft_reset() & self.soft_reset_mask != 0

    def initialise_core(self):
        super().initialise_core()
        if self._spin is not None:
            self._spin.on_reset(self)

    def next_wake_cycle(self, cycle_num):
        # A parked core is dormant exactly while every byte its loop can
        # observe still equals the parking snapshot; the check itself unparks
        # (and answers "next cycle") the moment anything differs, so a stride
        # can neither begin nor survive a change. See tt_sim/pe/rv/spin.py.
        if self.spin_parked:
            return self._spin.wake_check(self, cycle_num)
        return super().next_wake_cycle(cycle_num)

    def clock_tick(self, cycle_num):
        # These cores have a soft reset that they need to check
        is_in_reset = self._read_soft_reset() & self.soft_reset_mask != 0
        if not is_in_reset:
            if not self.soft_active:
                # About to be restarted, move into the resetted state
                self.initialise_core()
                self.soft_active = True
            spin = self._spin
            if spin is None:
                super().clock_tick(cycle_num)
            elif spin.state == SPIN_IDLE:
                super().clock_tick(cycle_num)
                if cycle_num >= spin.next_attempt:
                    spin.checkpoint(self, cycle_num)
            else:
                spin.tick(self, cycle_num)
        else:
            if self.soft_active:
                self.soft_active = False
                if self._spin is not None:
                    self._spin.on_reset(self)
