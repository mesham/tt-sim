from enum import IntEnum

from tt_sim.pe.rv.rv32 import RV32IM_TT
from tt_sim.pe.tensix.tensix_be_config_constants import (
    TensixBackendConfigurationConstants_ADDR32,
    TensixBackendConfigurationConstants_MASK,
    TensixBackendConfigurationConstants_SHAMT,
    tensix_be_config_parse_value,
)
from tt_sim.util.bits import get_nth_bit
from tt_sim.util.conversion import conv_to_uint32


class BabyRISCVCoreType(IntEnum):
    BRISC = 0
    NCRISC = 1
    TRISC0 = 2
    TRISC1 = 3
    TRISC2 = 4


class BabyRISCV(RV32IM_TT):
    CORE_TYPE_TO_SOFT_RESET_BIT = {
        BabyRISCVCoreType.BRISC: 11,
        BabyRISCVCoreType.NCRISC: 18,
        BabyRISCVCoreType.TRISC0: 12,
        BabyRISCVCoreType.TRISC1: 13,
        BabyRISCVCoreType.TRISC2: 14,
    }

    def __init__(self, core_type, memory_spaces, snoop=False):
        self.core_type = core_type
        self.soft_active = False
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
        super().__init__(start_addr, memory_spaces, [], snoop=snoop, core_id=core_id)

    def get_start_address(self):
        """
        Retrieves the start address of the Baby RISC-V core, this depends on which core it is.
        BRISC is always 0x0 which is the default, whereas the others have a default start address
        (already set) but this can be overridden by what is stored with the Tensix backend configuration
        https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/SoftReset.md
        """
        if self.core_type == BabyRISCVCoreType.BRISC:
            return self.start_address

        TENSIX_BACKEND_CONFIG_BASE = 0xFFEF_0000

        if self.core_type == BabyRISCVCoreType.NCRISC:
            override_tensix_config_addr_offset = (
                TensixBackendConfigurationConstants_ADDR32.NCRISC_RESET_PC_OVERRIDE_Reset_PC_Override_en
                * 4
            )
            override_tensix_config_shamt = TensixBackendConfigurationConstants_SHAMT.NCRISC_RESET_PC_OVERRIDE_Reset_PC_Override_en
            override_tensix_config_mask = TensixBackendConfigurationConstants_MASK.NCRISC_RESET_PC_OVERRIDE_Reset_PC_Override_en
            enabled_bit = 0
        else:
            override_tensix_config_addr_offset = (
                TensixBackendConfigurationConstants_ADDR32.TRISC_RESET_PC_OVERRIDE_Reset_PC_Override_en
                * 4
            )
            override_tensix_config_shamt = TensixBackendConfigurationConstants_SHAMT.TRISC_RESET_PC_OVERRIDE_Reset_PC_Override_en
            override_tensix_config_mask = TensixBackendConfigurationConstants_MASK.TRISC_RESET_PC_OVERRIDE_Reset_PC_Override_en
            enabled_bit = self.core_type - 2

        override_flag = conv_to_uint32(
            self.visible_memory.read(
                TENSIX_BACKEND_CONFIG_BASE + override_tensix_config_addr_offset, 4
            )
        )
        override_flag = tensix_be_config_parse_value(
            override_flag, override_tensix_config_shamt, override_tensix_config_mask
        )
        override_enabled = get_nth_bit(override_flag, enabled_bit)

        if override_enabled:
            if self.core_type == BabyRISCVCoreType.NCRISC:
                pc_val_addr_offset = (
                    TensixBackendConfigurationConstants_ADDR32.NCRISC_RESET_PC_PC * 4
                )
                pc_val_shamt = (
                    TensixBackendConfigurationConstants_SHAMT.NCRISC_RESET_PC_PC
                )
                pc_val_mask = (
                    TensixBackendConfigurationConstants_MASK.NCRISC_RESET_PC_PC
                )
            elif self.core_type == BabyRISCVCoreType.TRISC0:
                pc_val_addr_offset = (
                    TensixBackendConfigurationConstants_ADDR32.TRISC_RESET_PC_SEC0_PC
                    * 4
                )
                pc_val_shamt = (
                    TensixBackendConfigurationConstants_SHAMT.TRISC_RESET_PC_SEC0_PC
                )
                pc_val_mask = (
                    TensixBackendConfigurationConstants_MASK.TRISC_RESET_PC_SEC0_PC
                )
            elif self.core_type == BabyRISCVCoreType.TRISC1:
                pc_val_addr_offset = (
                    TensixBackendConfigurationConstants_ADDR32.TRISC_RESET_PC_SEC1_PC
                    * 4
                )
                pc_val_shamt = (
                    TensixBackendConfigurationConstants_SHAMT.TRISC_RESET_PC_SEC1_PC
                )
                pc_val_mask = (
                    TensixBackendConfigurationConstants_MASK.TRISC_RESET_PC_SEC1_PC
                )
            elif self.core_type == BabyRISCVCoreType.TRISC2:
                pc_val_addr_offset = (
                    TensixBackendConfigurationConstants_ADDR32.TRISC_RESET_PC_SEC2_PC
                    * 4
                )
                pc_val_shamt = (
                    TensixBackendConfigurationConstants_SHAMT.TRISC_RESET_PC_SEC2_PC
                )
                pc_val_mask = (
                    TensixBackendConfigurationConstants_MASK.TRISC_RESET_PC_SEC2_PC
                )
            else:
                raise Exception("Unknown core type")
            raw_pc = conv_to_uint32(
                self.visible_memory.read(
                    TENSIX_BACKEND_CONFIG_BASE + pc_val_addr_offset, 4
                )
            )
            return tensix_be_config_parse_value(raw_pc, pc_val_shamt, pc_val_mask)
        else:
            return self.start_address

    def clock_tick(self, cycle_num):
        # These cores have a soft reset that they need to check
        reset_val = conv_to_uint32(self.visible_memory.read(0xFFB121B0, 4))
        is_in_reset = (
            get_nth_bit(
                reset_val, BabyRISCV.CORE_TYPE_TO_SOFT_RESET_BIT[self.core_type]
            )
            == 1
        )
        if not is_in_reset:
            if not self.soft_active:
                # About to be restarted, move into the resetted state
                self.initialise_core()
                self.soft_active = True
            super().clock_tick(cycle_num)
        else:
            if self.soft_active:
                self.soft_active = False
