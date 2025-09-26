from tt_sim.pe.tensix.backends.backend_base import DataFormat, TensixBackendUnit
from tt_sim.pe.tensix.registers import LReg
from tt_sim.pe.tensix.util import DataFormatConversions
from tt_sim.util.bits import get_bits, get_nth_bit
from tt_sim.util.conversion import conv_to_float, conv_to_uint32


class VectorUnit(TensixBackendUnit):
    """
    SFPU vector unit, which has 32 lanes of 32 bit and 17 LRegs that can feed these lanes.

    This is based on the description and functional code snippets at
    https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/VectorUnit.md
    """

    OPCODE_TO_HANDLER = {
        "SFPENCC": "handle_sfpencc",
        "SFPLOADI": "handle_sfploadi",
        "SFPLOAD": "handle_sfpload",
        "SFPCONFIG": "handle_sfpconfig",
        "SFPIADD": "handle_sfpiadd",
        "SFPSTORE": "handle_sfpstore",
        "SFPMAD": "handle_mad",
        "SFPADD": "handle_add",
        "SFPMUL": "handle_mul",
        "SFPADDI": "handle_addi",
        "SFPMULI": "handle_muli",
        "SFPABS": "handle_sfpabs",
        "SFPSETSGN": "handle_sfpsetsgn",
        "SFPAND": "handle_sfpand",
        "SFPOR": "handle_sfpor",
        "SFPXOR": "handle_sfpxor",
        "SFPNOT": "handle_sfpnot",
        "SFPNOP": "handle_sfpnop",
        "SFPSETCC": "handle_sfpsetcc",
        "SFPPUSHC": "handle_sfppushc",
        "SFPPOPC": "handle_sfppopc",
        "SFPCOMPC": "handle_sfpcompc",
    }

    MOD1_IMM16_IS_VALUE = 1
    MOD1_BITWISE_OR = 2
    MOD1_BITWISE_AND = 4
    MOD1_BITWISE_XOR = 6
    MOD1_IMM16_IS_LANE_MASK = 8

    SFPLOADI_MOD0_FLOATB = 0  # Immediate is BF16
    SFPLOADI_MOD0_FLOATA = 1  # Immediate is FP16 (ish)
    SFPLOADI_MOD0_USHORT = 2  # Immediate is UINT16
    SFPLOADI_MOD0_SHORT = 4  # Immediate is INT16
    SFPLOADI_MOD0_UPPER = 8  # Immediate overwrites upper 16 bits
    SFPLOADI_MOD0_LOWER = 10  # Immediate overwrites lower 16 bits

    SFPENCC_MOD1_EC = 1  # Invert UseLaneFlagsForLaneEnable
    SFPENCC_MOD1_EI = 2  # Set UseLaneFlagsForLaneEnable from SFPENCC_IMM2_E
    SFPENCC_MOD1_RI = 8  # Set LaneFlags from SFPENCC_IMM2_R
    SFPENCC_IMM2_E = 1  # Immediate bit for UseLaneFlagsForLaneEnable
    SFPENCC_IMM2_R = 2  # Immediate bit for LaneFlags

    SFPSETCC_MOD1_IMM_BIT0 = 1
    SFPSETCC_MOD1_CLEAR = 8

    SFPSETCC_MOD1_LREG_LT0 = 0
    SFPSETCC_MOD1_LREG_NE0 = 2
    SFPSETCC_MOD1_LREG_GTE0 = 4
    SFPSETCC_MOD1_LREG_EQ0 = 6

    MOD0_FMT_SRCB = 0
    MOD0_FMT_FP16 = 1
    MOD0_FMT_BF16 = 2
    MOD0_FMT_FP32 = 3
    MOD0_FMT_INT32 = 4
    MOD0_FMT_INT8 = 5
    MOD0_FMT_UINT16 = 6
    MOD0_FMT_HI16 = 7
    MOD0_FMT_INT16 = 8
    MOD0_FMT_LO16 = 9
    MOD0_FMT_INT32_ALL = 10
    MOD0_FMT_ZERO = 11
    MOD0_FMT_INT32_SM = 12
    MOD0_FMT_INT8_COMP = 13
    MOD0_FMT_LO16_ONLY = 14
    MOD0_FMT_HI16_ONLY = 15

    SFPIADD_MOD1_ARG_LREG_DST = 0
    SFPIADD_MOD1_ARG_IMM = 1
    SFPIADD_MOD1_ARG_2SCOMP_LREG_DST = 2
    SFPIADD_MOD1_CC_LT0 = 0
    SFPIADD_MOD1_CC_NONE = 4
    SFPIADD_MOD1_CC_GTE0 = 8
    SFPMAD_MOD1_INDIRECT_VA = 4
    SFPMAD_MOD1_INDIRECT_VD = 8
    SFPABS_MOD1_FLOAT = 1
    SFPSETSGN_MOD1_ARG_IMM = 1

    ENABLE_FP16A_INF = (0, 1)
    DISABLE_BACKDOOR_LOAD = (1, 1)
    ENABLE_DEST_INDEX = (2, 1)
    CAPTURE_DEFAULT_DEST_INDEX = (3, 1)
    BLOCK_DEST_WR_FROM_SFPU = (4, 1)
    BLOCK_SFPU_RD_FROM_DEST = (5, 1)
    DEST_RD_COL_EXCHANGE = (6, 1)
    DEST_WR_COL_EXCHANGE = (7, 1)
    EXCHANGE_SRCB_SRCC = (8, 1)
    BLOCK_DEST_MOV = (9, 2)
    ROW_MASK = (12, 4)

    class LoadMacroConfig:
        def __init__(self):
            self.storeMod0 = 0
            self.usesLoadMod0ForStore = False
            self.unitDelayKind = 0
            self.sequence = [0] * 4
            self.instructionTemplate = [0] * 4

        def misc(self, value, mode=0):
            self.storeMod0 = get_bits(value, 0, 3)
            self.usesLoadMod0ForStore = get_bits(value, 4, 7)
            self.unitDelayKind = get_bits(value, 8, 11)
            if mode == VectorUnit.MOD1_BITWISE_OR:
                self.storeMod0 |= self.storeMod0
                self.usesLoadMod0ForStore |= self.usesLoadMod0ForStore
                self.unitDelayKind |= self.unitDelayKind
            elif mode == VectorUnit.MOD1_BITWISE_AND:
                self.storeMod0 &= self.storeMod0
                self.usesLoadMod0ForStore &= self.usesLoadMod0ForStore
                self.unitDelayKind &= self.unitDelayKind
            elif mode == VectorUnit.MOD1_BITWISE_XOR:
                self.storeMod0 ^= self.storeMod0
                self.usesLoadMod0ForStore ^= self.usesLoadMod0ForStore
                self.unitDelayKind ^= self.unitDelayKind

    def __init__(self, backend):
        self.lregs = [LReg() for i in range(17)]
        self.lregs[8].setReadOnly(0.8373)
        self.lregs[9].setReadOnly(0)
        self.lregs[10].setReadOnly(1.0)
        for i in range(32):
            self.lregs[15][i] = i * 2
        self.lregs[15].setReadOnly()

        self.laneFlags = [False] * 32
        self.useLaneFlagsForLaneEnable = [False] * 32
        self.flagStack = []
        self.laneConfig = [0] * 32
        self.loadMacroConfig = [VectorUnit.LoadMacroConfig() for i in range(32)]
        super().__init__(backend, VectorUnit.OPCODE_TO_HANDLER, "Vector")

    def laneConfigValue(self, lane, key):
        assert len(key) == 2
        return get_bits(self.laneConfig[lane], key[0], (key[0] + key[1]) - 1)

    def handle_sfpnot(self, instruction_info, issue_thread, instr_args):
        vd = instr_args["lreg_dest"]
        vc = instr_args["lreg_c"]

        if self.getDiagnosticSettings().reportSFPUCalculations():
            print(f"SFPU: lreg[{vd}] = ~ lreg[{vc}]")

        if vd < 8 or vd == 16:
            for lane in range(32):
                if self.isLaneEnabled(lane):
                    self.lregs[vd][lane] = ~self.lregs[vc][lane]

    def handle_sfpxor(self, instruction_info, issue_thread, instr_args):
        vd = instr_args["lreg_dest"]
        vc = instr_args["lreg_c"]
        vb = vd

        if self.getDiagnosticSettings().reportSFPUCalculations():
            print(f"SFPU: lreg[{vd}] = lreg[{vb}] ^ lreg[{vc}]")
        if vd < 8 or vd == 16:
            for lane in range(32):
                if self.isLaneEnabled(lane):
                    self.lregs[vd][lane] = self.lregs[vb][lane] ^ self.lregs[vc][lane]

    def handle_sfpand(self, instruction_info, issue_thread, instr_args):
        vd = instr_args["lreg_dest"]
        vc = instr_args["lreg_c"]
        vb = vd

        if self.getDiagnosticSettings().reportSFPUCalculations():
            print(f"SFPU: lreg[{vd}] = lreg[{vb}] & lreg[{vc}]")
        if vd < 8 or vd == 16:
            for lane in range(32):
                if self.isLaneEnabled(lane):
                    self.lregs[vd][lane] = self.lregs[vb][lane] & self.lregs[vc][lane]

    def handle_sfpor(self, instruction_info, issue_thread, instr_args):
        vd = instr_args["lreg_dest"]
        vc = instr_args["lreg_c"]
        vb = vd

        if self.getDiagnosticSettings().reportSFPUCalculations():
            print(f"SFPU: lreg[{vd}] = lreg[{vb}] | lreg[{vc}]")

        if vd < 8 or vd == 16:
            for lane in range(32):
                if self.isLaneEnabled(lane):
                    self.lregs[vd][lane] = self.lregs[vb][lane] | self.lregs[vc][lane]

    def handle_sfpsetsgn(self, instruction_info, issue_thread, instr_args):
        mod1 = instr_args["instr_mod1"]
        vd = instr_args["lreg_dest"]
        vc = instr_args["lreg_c"]
        imm1 = instr_args["imm12_math"]

        vb = vd
        if vd < 8 or vd == 16:
            for lane in range(32):
                if self.isLaneEnabled(lane):
                    c = self.lregs[vc][lane]
                    exp = (c >> 23) & 0xFF
                    man = c & 0x7FFFFF
                    if mod1 & VectorUnit.SFPSETSGN_MOD1_ARG_IMM:
                        sign = imm1 & 0x1
                    else:
                        b = self.lregs[vb][lane]
                        sign = b >> 31
                    self.lregs[vd][lane] = (sign << 31) | (exp << 23) | man

    def handle_sfpabs(self, instruction_info, issue_thread, instr_args):
        mod1 = instr_args["instr_mod1"]
        vd = instr_args["lreg_dest"]
        vc = instr_args["lreg_c"]

        if self.getDiagnosticSettings().reportSFPUCalculations():
            print(f"SFPU: lreg[{vd}] = abs(lreg[{vc}])")

        if vd < 8 or vd == 16:
            for lane in range(32):
                if self.isLaneEnabled(lane):
                    x = self.lregs[vc][lane]
                    if x >= 0x80000000:
                        # Sign bit is set, i.e. value is negative.
                        if mod1 & VectorUnit.SFPABS_MOD1_FLOAT:
                            if x > 0xFF800000:
                                # Value is -NaN; leave it as -NaN
                                pass
                            else:
                                # Clear the sign bit, i.e. floating-point negation
                                x &= 0x7FFFFFFF
                        else:
                            # Two's complement integer negation, unless the input is
                            # -2147483648, in which case it remains as -2147483648
                            x = -x
                    else:
                        # Value is positive (or zero); leave it as-is
                        pass
                    self.lregs[vd][lane] = x

    def handle_addi(self, instruction_info, issue_thread, instr_args):
        mod1 = instr_args["instr_mod1"]
        vd = instr_args["lreg_dest"]
        imm16 = instr_args["imm16_math"]
        vc = vd

        if self.getDiagnosticSettings().reportSFPUCalculations():
            print(f"SFPU: lreg[{vd}] = {hex(imm16)} + lreg[{vc}]")
        for lane in range(32):
            if vd < 12 or self.laneConfigValue(lane, VectorUnit.DISABLE_BACKDOOR_LOAD):
                if self.isLaneEnabled(lane):
                    c = self.lregs[vc][lane]
                    d = self.BF16ToFP32(imm16) + c
                    if (mod1 & VectorUnit.SFPMAD_MOD1_INDIRECT_VD) and vd != 16:
                        vd = self.lregs[7][lane] & 15
                    else:
                        vd = vd
                    if vd < 8 or vd == 16:
                        self.lregs[vd][lane] = d

    def handle_muli(self, instruction_info, issue_thread, instr_args):
        mod1 = instr_args["instr_mod1"]
        vd = instr_args["lreg_dest"]
        imm16 = instr_args["imm16_math"]
        vc = vd

        if self.getDiagnosticSettings().reportSFPUCalculations():
            print(f"SFPU: lreg[{vd}] = {hex(imm16)} * lreg[{vc}]")

        for lane in range(32):
            if vd < 12 or self.laneConfigValue(lane, VectorUnit.DISABLE_BACKDOOR_LOAD):
                if self.isLaneEnabled(lane):
                    c = self.lregs[vc][lane]
                    d = self.BF16ToFP32(imm16) * c
                    if (mod1 & VectorUnit.SFPMAD_MOD1_INDIRECT_VD) and vd != 16:
                        vd = self.lregs[7][lane] & 15
                    else:
                        vd = vd
                    if vd < 8 or vd == 16:
                        self.lregs[vd][lane] = d

    def perform_mad(self, va, vb, vc, vd, mod1):
        if self.getDiagnosticSettings().reportSFPUCalculations():
            print(f"SFPU: lreg[{vd}] = lreg[{va}] * lreg[{vb}] + lreg[{vc}]")
        for lane in range(32):
            if vd < 12 or self.laneConfigValue(lane, VectorUnit.DISABLE_BACKDOOR_LOAD):
                if self.isLaneEnabled(lane):
                    va = (
                        self.lregs[7][lane] & 15
                        if mod1 & VectorUnit.SFPMAD_MOD1_INDIRECT_VA
                        else va
                    )
                    a = self.lregs[va][lane]
                    b = self.lregs[vb][lane]
                    c = self.lregs[vc][lane]
                    d = a * b + c
                    if (mod1 & VectorUnit.SFPMAD_MOD1_INDIRECT_VD) and vd != 16:
                        vd = self.lregs[7][lane] & 15
                    else:
                        vd = vd
                    if vd < 8 or vd == 16:
                        self.lregs[vd][lane] = d

    def handle_mad(self, instruction_info, issue_thread, instr_args):
        mod1 = instr_args["instr_mod1"]
        vd = instr_args["lreg_dest"]
        vc = instr_args["lreg_src_c"]
        vb = instr_args["lreg_src_b"]
        va = instr_args["lreg_src_a"]

        self.perform_mad(va, vb, vc, vd, mod1)

    def handle_add(self, instruction_info, issue_thread, instr_args):
        mod1 = instr_args["instr_mod1"]
        vd = instr_args["lreg_dest"]
        vc = instr_args["lreg_src_c"]
        vb = instr_args["lreg_src_b"]
        va = instr_args["lreg_src_a"]

        va = 10  # hardcoded to be lanes containing 1.0
        self.perform_mad(va, vb, vc, vd, mod1)

    def handle_mul(self, instruction_info, issue_thread, instr_args):
        mod1 = instr_args["instr_mod1"]
        vd = instr_args["lreg_dest"]
        vc = instr_args["lreg_src_c"]
        vb = instr_args["lreg_src_b"]
        va = instr_args["lreg_src_a"]

        vc = 9  # hardcoded to be lanes containing 0
        self.perform_mad(va, vb, vc, vd, mod1)

    def handle_sfpcompc(self, instruction_info, issue_thread, instr_args):
        vd = instr_args["lreg_dest"]

        do_compc = False
        for lane in range(32):
            if vd < 12 or self.laneConfigValue(lane, VectorUnit.DISABLE_BACKDOOR_LOAD):
                do_compc = True
                break

        if do_compc:
            if len(self.flagStack) == 0:
                top = (True, True)
            else:
                top = self.flagStack[0]

            # Note we are doing this on a lane by lane basis, whereas have implemented
            # popc and pushc across all lanes
            for lane in range(32):
                # Invert laneFlags, subject to top
                if top[1][lane] and self.useLaneFlagsForLaneEnable[lane]:
                    self.laneFlags[lane] = top[0][lane] and (not self.laneFlags[lane])
                else:
                    self.laneFlags[lane] = False

    def handle_sfppopc(self, instruction_info, issue_thread, instr_args):
        mod1 = instr_args["instr_mod1"]
        vd = instr_args["lreg_dest"]

        do_pop = False
        for lane in range(32):
            if vd < 12 or self.laneConfigValue(lane, VectorUnit.DISABLE_BACKDOOR_LOAD):
                do_pop = True
                break

        if do_pop:
            if len(self.flagStack) == 0:
                top = (False, False)
            else:
                top = self.flagStack[0]

            if mod1 == 0:
                # Plain pop from stack
                assert len(self.flagStack) > 0
                self.flagStack.pop(0)
            elif len(self.flagStack) == 8:
                self.flagStack[7] = top

            if mod1 == 0:
                # Set LaneFlags and UseLaneFlagsForLaneEnable to Top
                self.laneFlags = top[0]
                self.useLaneFlagsForLaneEnable = top[1]
            elif mod1 <= 12:
                # Mutate LaneFlags and UseLaneFlagsForLaneEnable based on Top
                self.laneFlags = list(self.booleanOp(mod1, self.laneFlags, top[0]))
                self.useLaneFlagsForLaneEnable = top[1]
            elif mod1 == 13:
                # Just invert laneFlags
                self.laneFlags = [not v for v in self.laneFlags]
            elif mod1 == 14:
                # Set laneFlags and useLaneFlagsForLaneEnable to constants
                self.laneFlags = self.useLaneFlagsForLaneEnable = [
                    True for _ in range(32)
                ]
            elif mod1 == 15:
                # Set LaneFlags and UseLaneFlagsForLaneEnable to constants
                self.useLaneFlagsForLaneEnable = [True for _ in range(32)]
                self.laneFlags = [False for _ in range(32)]

    def booleanOp(self, mod1, A_list, B_list):
        for A, B in zip(A_list, B_list):
            match mod1:
                case 1:
                    yield B
                case 2:
                    yield not B
                case 3:
                    yield A and B
                case 4:
                    yield A or B
                case 5:
                    yield A and (not B)
                case 6:
                    yield A or (not B)
                case 7:
                    yield (not A) and B
                case 8:
                    yield (not A) or B
                case 9:
                    yield (not A) and (not B)
                case 10:
                    yield (not A) or (not B)
                case 11:
                    yield A != B
                case 12:
                    yield A == B

    def handle_sfppushc(self, instruction_info, issue_thread, instr_args):
        vd = instr_args["lreg_dest"]
        do_push = False
        for lane in range(32):
            if vd < 12 or self.laneConfigValue(lane, VectorUnit.DISABLE_BACKDOOR_LOAD):
                do_push = True
                break

        if do_push:
            assert self.flagStack.Size() < 8
            self.flagStack.append((self.laneFlags, self.useLaneFlagsForLaneEnable))

    def handle_sfpsetcc(self, instruction_info, issue_thread, instr_args):
        mod1 = instr_args["instr_mod1"]
        vd = instr_args["lreg_dest"]
        vc = instr_args["lreg_c"]
        imm1 = instr_args["imm12_math"] & 0x1

        for lane in range(32):
            if vd < 12 or self.laneConfigValue(lane, VectorUnit.DISABLE_BACKDOOR_LOAD):
                if self.isLaneEnabled(
                    lane
                ):  # Is this correct? Seems strange that can not reenable
                    if not self.useLaneFlagsForLaneEnable[lane]:
                        self.laneFlags[lane] = False
                    elif mod1 & VectorUnit.SFPSETCC_MOD1_CLEAR:
                        self.laneFlags[lane] = False
                    elif mod1 & VectorUnit.SFPSETCC_MOD1_IMM_BIT0:
                        self.laneFlags[lane] = imm1 != 0
                    else:
                        c = self.lregs[vc][lane]
                        match mod1:
                            case VectorUnit.SFPSETCC_MOD1_LREG_LT0:
                                self.laneFlags[lane] = c < 0
                            case VectorUnit.SFPSETCC_MOD1_LREG_NE0:
                                self.laneFlags[lane] = c != 0
                            case VectorUnit.SFPSETCC_MOD1_LREG_GTE0:
                                self.laneFlags[lane] = c >= 0
                            case VectorUnit.SFPSETCC_MOD1_LREG_EQ0:
                                self.laneFlags[lane] = c == 0

    def handle_sfpencc(self, instruction_info, issue_thread, instr_args):
        mod1 = instr_args["instr_mod1"]
        vd = instr_args["lreg_dest"]
        imm2 = instr_args["imm12_math"]

        for lane in range(32):
            if vd < 12 or self.laneConfigValue(lane, VectorUnit.DISABLE_BACKDOOR_LOAD):
                if mod1 & VectorUnit.SFPENCC_MOD1_EI:
                    self.useLaneFlagsForLaneEnable[lane] = (
                        imm2 & VectorUnit.SFPENCC_IMM2_E
                    ) != 0
                elif mod1 & VectorUnit.SFPENCC_MOD1_EC:
                    self.useLaneFlagsForLaneEnable[
                        lane
                    ] = not self.useLaneFlagsForLaneEnable[lane]
                else:
                    # UseLaneFlagsForLaneEnable left as-is.
                    pass

                if mod1 & VectorUnit.SFPENCC_MOD1_RI:
                    self.laneFlags[lane] = (imm2 & VectorUnit.SFPENCC_IMM2_R) != 0
                else:
                    self.laneFlags[lane] = True

    def handle_sfpiadd(self, instruction_info, issue_thread, instr_args):
        mod1 = instr_args["instr_mod1"]
        vd = instr_args["lreg_dest"]
        vc = instr_args["lreg_c"]
        imm12 = instr_args["imm12_math"] & 0xFFF

        vb = vd

        if self.getDiagnosticSettings().reportSFPUCalculations():
            print(f"SFPU: lreg[{vd}] = lreg[{vc}] + lreg[{vb}]")

        if vd < 8 or vd == 16:
            for lane in range(32):
                if self.isLaneEnabled(lane):
                    if mod1 & VectorUnit.SFPIADD_MOD1_ARG_IMM:
                        self.lregs[vd][lane] = self.lregs[vc][lane] + imm12
                    elif mod1 & VectorUnit.SFPIADD_MOD1_ARG_2SCOMP_LREG_DST:
                        self.lregs[vd][lane] = (
                            self.lregs[vc][lane] - self.lregs[vb][lane]
                        )
                    else:
                        self.lregs[vd][lane] = (
                            self.lregs[vc][lane] + self.lregs[vb][lane]
                        )

                    if vd < 8:
                        if mod1 & VectorUnit.SFPIADD_MOD1_CC_NONE:
                            # Leave LaneFlags as-is
                            pass
                        else:
                            self.laneFlags[lane] = self.lregs[vd][lane] < 0

                        if mod1 & VectorUnit.SFPIADD_MOD1_CC_GTE0:
                            self.laneFlags[lane] = not self.laneFlags[lane]

    def get_dst_address(self, issue_thread, mod0, imm10):
        stateID = self.backend.getThreadConfigValue(
            issue_thread, "CFG_STATE_ID_StateID"
        )

        if mod0 == VectorUnit.MOD0_FMT_SRCB:
            if self.getConfigValue(stateID, "ALU_ACC_CTRL_SFPU_Fp32_enabled"):
                # Functionally identical to MOD0_FMT_INT32
                mod0 = VectorUnit.MOD0_FMT_FP32
            else:
                srcBFmt = (
                    self.getConfigValue(stateID, "ALU_FORMAT_SPEC_REG_SrcB_val")
                    if self.getConfigValue(stateID, "ALU_FORMAT_SPEC_REG_SrcB_override")
                    else self.getConfigValue(stateID, "ALU_FORMAT_SPEC_REG1_SrcB")
                )
                if srcBFmt in [
                    DataFormat.FP32,
                    DataFormat.TF32,
                    DataFormat.BF16,
                    DataFormat.BF16,
                    DataFormat.BFP4,
                    DataFormat.BFP2,
                    DataFormat.INT32,
                    DataFormat.UINT16,
                ]:
                    mod0 = VectorUnit.MOD0_FMT_BF16
                else:
                    mod0 = VectorUnit.MOD0_FMT_FP16

        # Apply various Dst address adjustments.
        # The top 8 bits of Addr end up selecting an aligned group of four rows of Dst, the
        # next bit selects between even and odd columns, and the low bit goes unused.

        addr = imm10 + self.backend.getThreadConfigValue(
            issue_thread, "DEST_TARGET_REG_CFG_MATH_Offset"
        )
        if mod0 == VectorUnit.MOD0_FMT_INT32_ALL:
            addr += (
                self.backend.getRWC(issue_thread).Dst
                + self.getConfigValue(stateID, "DEST_REGW_BASE_Base")
                & 3
            )
        else:
            addr += self.backend.getRWC(issue_thread).Dst + self.getConfigValue(
                stateID, "DEST_REGW_BASE_Base"
            )

        return addr, mod0

    def handle_sfpstore(self, instruction_info, issue_thread, instr_args):
        imm10 = instr_args["dest_reg_addr"]
        addrmod = instr_args["sfpu_addr_mode"]
        mod0 = instr_args["instr_mod0"]
        vd = instr_args["lreg_ind"]

        addr, mod0 = self.get_dst_address(issue_thread, mod0, imm10)

        if self.getDiagnosticSettings().reportSFPUCalculations():
            if addr & 2:
                col_start = 1
            else:
                col_start = 0
            print(
                f"SFPU: store lreg[{vd}] into between dst[{(addr & ~3)}, {col_start}] and dst"
                f"[{(addr & ~3) + int(31 / 8)}, X] from thread{issue_thread}"
            )

        for lane in range(32):
            if self.laneConfigValue(lane, VectorUnit.BLOCK_SFPU_RD_FROM_DEST):
                continue
            if vd < 12 or self.laneConfigValue(lane, VectorUnit.DISABLE_BACKDOOR_LOAD):
                if self.isLaneEnabled(lane) or mod0 == VectorUnit.MOD0_FMT_INT32_ALL:
                    row = (addr & ~3) + int(lane / 8)
                    column = (lane & 7) * 2
                    if addr & 2 or self.laneConfigValue(
                        lane & 7, VectorUnit.DEST_RD_COL_EXCHANGE
                    ):
                        column += 1

                    datum = self.lregs[vd][lane]
                    match mod0:
                        case VectorUnit.MOD0_FMT_FP16:
                            write_val = DataFormatConversions.FP16ToDstFormatFP16(
                                DataFormatConversions.FP32ToFP16(conv_to_uint32(datum))
                            )
                            self.getDst().setDst16b(row, column, write_val)
                        case VectorUnit.MOD0_FMT_BF16:
                            write_val = DataFormatConversions.BF16ToDstFormatBF16(
                                DataFormatConversions.FP32ToBF16(conv_to_uint32(datum))
                            )
                            self.getDst().setDst16b(row, column, write_val)
                        case (
                            VectorUnit.MOD0_FMT_FP32
                            | VectorUnit.MOD0_FMT_INT32
                            | VectorUnit.MOD0_FMT_INT32_ALL
                        ):
                            self.getDst().setDst32b(
                                row,
                                column,
                                DataFormatConversions.FP32ToDstFormatFP32(
                                    conv_to_uint32(datum)
                                ),
                            )
                        case VectorUnit.MOD0_FMT_INT32_SM:
                            write_val = DataFormatConversions.FP32ToDstFormatFP32(
                                DataFormatConversions.toSignMag(datum)
                            )
                            self.getDst().setDst32b(row, column, write_val)
                        case VectorUnit.MOD0_FMT_INT8:
                            write_val = DataFormatConversions.FP16ToDstFormatFP16(
                                DataFormatConversions.signMag11ToFP16(datum)
                            )
                            self.getDst().setDst16b(row, column, write_val)
                        case VectorUnit.MOD0_FMT_INT8_COMP:
                            write_val = DataFormatConversions.FP16ToDstFormatFP16(
                                DataFormatConversions.signMag11ToFP16(
                                    DataFormatConversions.ToSignMag(datum)
                                )
                            )
                            self.getDst().setDst16b(row, column, write_val)
                        case VectorUnit.MOD0_FMT_LO16_ONLY | VectorUnit.MOD0_FMT_UINT16:
                            self.getDst().setDst16b(row, column, datum & 0xFFFF)
                        case VectorUnit.MOD0_FMT_HI16_ONLY:
                            self.getDst().setDst16b(row, column, datum >> 16)
                        case VectorUnit.MOD0_FMT_INT16:
                            self.getDst().setDst16b(
                                row, column, ((datum >> 31) << 15) | (datum & 0x7FFF)
                            )
                        case VectorUnit.MOD0_FMT_LO16:
                            self.getDst().setDst32b(
                                row, column, (datum << 16) | (datum >> 16)
                            )
                        case VectorUnit.MOD0_FMT_HI16:
                            self.getDst().setDst32b(row, column, datum)
                        case VectorUnit.MOD0_FMT_ZERO:
                            self.getDst().setDst16b(row, column, 0)
                        case _:
                            raise NotImplementedError()

        self.backend.getRWC(issue_thread).applyPartialAddrMod(issue_thread, addrmod)

    def handle_sfpload(self, instruction_info, issue_thread, instr_args):
        imm10 = instr_args["dest_reg_addr"]
        addrmod = instr_args["sfpu_addr_mode"]
        mod0 = instr_args["instr_mod0"]
        vd = instr_args["lreg_ind"]

        addr, mod0 = self.get_dst_address(issue_thread, mod0, imm10)

        if self.getDiagnosticSettings().reportSFPUCalculations():
            if addr & 2:
                col_start = 1
            else:
                col_start = 0
            print(
                f"SFPU: load between dst[{(addr & ~3)}, {col_start}] and dst[{(addr & ~3) + int(31 / 8)}, "
                f"X]into lreg[{vd}] from thread{issue_thread}"
            )

        if vd < 8:
            for lane in range(32):
                if self.laneConfigValue(lane, VectorUnit.BLOCK_SFPU_RD_FROM_DEST):
                    continue
                if self.isLaneEnabled(lane) or mod0 == VectorUnit.MOD0_FMT_INT32_ALL:
                    row = (addr & ~3) + int(lane / 8)
                    column = (lane & 7) * 2
                    if addr & 2 or self.laneConfigValue(
                        lane & 7, VectorUnit.DEST_RD_COL_EXCHANGE
                    ):
                        column += 1

                    match mod0:
                        case VectorUnit.MOD0_FMT_FP16:
                            rd = self.getDst().getDst16b(row, column)
                            datum = conv_to_float(
                                DataFormatConversions.FP16InDstToFP32(
                                    rd,
                                    self.laneConfigValue(
                                        lane, VectorUnit.ENABLE_FP16A_INF
                                    ),
                                )
                            )
                        case VectorUnit.MOD0_FMT_BF16:
                            rd = self.getDst().getDst16b(row, column)
                            datum = conv_to_float(
                                DataFormatConversions.BF16InDstToBF16(rd) << 16
                            )
                        case VectorUnit.MOD0_FMT_FP32:
                            rd = self.getDst().getDst32b(row, column)
                            datum = conv_to_float(
                                DataFormatConversions.FP32InDstToFP32(rd)
                            )
                        case VectorUnit.MOD0_FMT_INT32 | VectorUnit.MOD0_FMT_INT32_ALL:
                            rd = self.getDst().getDst32b(row, column)
                            datum = DataFormatConversions.FP32InDstToFP32(rd)
                        case VectorUnit.MOD0_FMT_INT32_SM:
                            rd = self.getDst().getDst32b(row, column)
                            datum = DataFormatConversions.signMagToTwosComp(
                                DataFormatConversions.FP32InDstToFP32(rd)
                            )
                        case VectorUnit.MOD0_FMT_INT8:
                            rd = self.getDst().getDst16b(row, column)
                            datum = DataFormatConversions.signMag8ToSignMag32(rd)
                        case VectorUnit.MOD0_FMT_INT8_COMP:
                            rd = self.getDst().getDst16b(row, column)
                            datum = DataFormatConversions.signMagToTwosComp(
                                DataFormatConversions.signMag11ToSignMag32(rd)
                            )
                        case VectorUnit.MOD0_FMT_LO16_ONLY:
                            rd = self.getDst().getDst16b(row, column)
                            datum = (self.lregs[vd][lane] & 0xFFFF0000) | rd
                        case VectorUnit.MOD0_FMT_HI16_ONLY:
                            rd = self.getDst().getDst16b(row, column)
                            datum = (rd << 16) | (self.lregs[vd][lane] & 0xFFFF)
                        case VectorUnit.MOD0_FMT_HI16_ONLY:
                            rd = self.getDst().getDst16b(row, column)
                            datum = DataFormatConversions.signMag16ToSignMag32(rd)
                        case VectorUnit.MOD0_FMT_UINT16 | VectorUnit.MOD0_FMT_LO16:
                            datum = DataFormatConversions.signMag16ToSignMag32(rd)
                        case VectorUnit.MOD0_FMT_HI16:
                            datum = DataFormatConversions.signMag16ToSignMag32(rd) << 16
                        case VectorUnit.MOD0_FMT_ZERO:
                            datum = 0
                        case _:
                            raise NotImplementedError()

                    self.lregs[vd][lane] = datum
                    if (
                        (vd < 4)
                        and self.laneConfigValue(lane, VectorUnit.ENABLE_DEST_INDEX)
                        and self.laneConfigValue(
                            lane, VectorUnit.CAPTURE_DEFAULT_DEST_INDEX
                        )
                    ):
                        self.lregs[vd + 4][lane] = (row << 4) | column

        self.backend.getRWC(issue_thread).applyPartialAddrMod(issue_thread, addrmod)

    def handle_sfploadi(self, instruction_info, issue_thread, instr_args):
        mod0 = instr_args["instr_mod0"]
        vd = instr_args["lreg_ind"]
        imm16 = instr_args["imm16"]

        assert vd < 8
        for lane in range(32):
            if self.isLaneEnabled(lane):
                match mod0:
                    case VectorUnit.SFPLOADI_MOD0_FLOATB:
                        self.lregs[vd][lane] = self.BF16toFP32(imm16)
                    case VectorUnit.SFPLOADI_MOD0_FLOATA:
                        self.lregs[vd][lane] = self.FP16toFP32(imm16)
                    case VectorUnit.SFPLOADI_MOD0_USHORT:
                        self.lregs[vd][lane] = imm16
                    case VectorUnit.SFPLOADI_MOD0_SHORT:
                        self.lregs[vd][lane] = imm16
                    case VectorUnit.SFPLOADI_MOD0_UPPER:
                        self.lregs[vd][lane] = (imm16 << 16) | (
                            self.lregs[vd][lane] & 0x0000FFFF
                        )
                    case VectorUnit.SFPLOADI_MOD0_LOWER:
                        self.lregs[vd][lane] = (
                            self.lregs[vd][lane] & 0xFFFF0000
                        ) | imm16
                    case _:
                        raise ValueError()

    def handle_sfpnop(self, instruction_info, issue_thread, instr_args):
        pass

    def handle_sfpconfig(self, instruction_info, issue_thread, instr_args):
        mod1 = instr_args["instr_mod1"]
        vd = instr_args["config_dest"]
        imm16 = instr_args["imm16_math"]
        for lane in range(32):
            if mod1 & VectorUnit.MOD1_IMM16_IS_LANE_MASK:
                if not get_nth_bit(imm16, (lane & 7) * 2):
                    continue

            if self.useLaneFlagsForLaneEnable[lane & 7]:
                if not self.laneFlags[lane & 7]:
                    continue

            match vd:
                case 0 | 1 | 2 | 3:
                    # Write to LoadMacroConfig::InstructionTemplate.
                    self.loadMacroConfig[lane].instructionTemplate[vd] = self.lregs[0][
                        lane & 7
                    ]
                case 4 | 5 | 6 | 7:
                    # Write to LoadMacroConfig::Sequence
                    value = (
                        imm16
                        if (mod1 & VectorUnit.MOD1_IMM16_IS_VALUE)
                        else self.lregs[0][lane & 7]
                    )
                    self.loadMacroConfig[lane].sequence[vd - 4] = value
                case 8:
                    # Write or manipulate LoadMacroConfig::Misc
                    value = (
                        imm16
                        if (mod1 & VectorUnit.MOD1_IMM16_IS_VALUE)
                        else self.lregs[0][lane & 7]
                    )
                    self.loadMacroConfig[lane].misc(value, mod1 & 6)
                case 9 | 10:
                    # Does nothing
                    pass
                case 11 | 12 | 13 | 14:
                    if mod1 & VectorUnit.MOD1_IMM16_IS_VALUE:
                        match vd:
                            case 11:
                                value = -1.0
                            case 12:
                                value = -1 / 65536.0
                            case 13:
                                value = -0.67487759
                            case 14:
                                value = -0.34484843
                    else:
                        value = self.lregs[0][lane & 7]
                    self.lregs[vd][lane] = value
                case 15:
                    # Write or manipulate LaneConfig
                    original = self.laneConfig[lane]
                    value = (
                        imm16
                        if (mod1 & VectorUnit.MOD1_IMM16_IS_VALUE)
                        else self.lregs[0][lane & 7]
                    )
                    match mod1 & 6:
                        case 0:
                            self.laneConfig[lane] = value
                        case VectorUnit.MOD1_BITWISE_OR:
                            self.laneConfig[lane] |= value
                        case VectorUnit.MOD1_BITWISE_AND:
                            self.laneConfig[lane] &= value
                        case VectorUnit.MOD1_BITWISE_XOR:
                            self.laneConfig[lane] ^= value

                    if mod1 & VectorUnit.MOD1_IMM16_IS_VALUE:
                        self.laneConfig[lane] |= original & ~0xFFFF

    def isLaneEnabled(self, lane):
        if get_nth_bit(
            self.laneConfigValue(lane & 7, VectorUnit.ROW_MASK), int(lane / 8)
        ):
            return False
        elif self.useLaneFlagsForLaneEnable[lane]:
            return self.laneFlags[lane]
        else:
            return True

    def BF16toFP32(self, val):
        return val << 16

    def FP16toFP32(self, val):
        sign = val >> 15
        exp = (val >> 10) & 0x1F
        man = val & 0x3FF

        exp += 112  # Rebias 5b exponent to 8b
        return (sign << 31) | (exp << 23) | (man << 13)
