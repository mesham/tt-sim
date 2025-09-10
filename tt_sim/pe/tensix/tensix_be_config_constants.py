from enum import IntEnum


class TensixBackendConfigurationConstants_ADDR32(IntEnum):
    TRISC_RESET_PC_SEC0_PC = 158
    TRISC_RESET_PC_SEC1_PC = 159
    TRISC_RESET_PC_SEC2_PC = 160
    TRISC_RESET_PC_OVERRIDE_Reset_PC_Override_en = 161
    NCRISC_RESET_PC_PC = 162
    NCRISC_RESET_PC_OVERRIDE_Reset_PC_Override_en = 163


class TensixBackendConfigurationConstants_SHAMT(IntEnum):
    TRISC_RESET_PC_SEC0_PC = 0
    TRISC_RESET_PC_SEC1_PC = 0
    TRISC_RESET_PC_SEC2_PC = 0
    TRISC_RESET_PC_OVERRIDE_Reset_PC_Override_en = 0
    NCRISC_RESET_PC_PC = 0
    NCRISC_RESET_PC_OVERRIDE_Reset_PC_Override_en = 0


class TensixBackendConfigurationConstants_MASK(IntEnum):
    TRISC_RESET_PC_SEC0_PC = 0xFFFFFFFF
    TRISC_RESET_PC_SEC1_PC = 0xFFFFFFFF
    TRISC_RESET_PC_SEC2_PC = 0xFFFFFFFF
    TRISC_RESET_PC_OVERRIDE_Reset_PC_Override_en = 0x7
    NCRISC_RESET_PC_PC = 0xFFFFFFFF
    NCRISC_RESET_PC_OVERRIDE_Reset_PC_Override_en = 0x1


def tensix_be_config_parse_value(value, shamt, mask):
    return (value & mask) >> shamt
