import os

import numpy as np

from tt_sim.pe.tensix.backends.backend_base import DataFormat, TensixBackendUnit
from tt_sim.pe.tensix.backends.fpu_jit import get_fused_mvmul
from tt_sim.pe.tensix.backends.vector import VectorUnit
from tt_sim.pe.tensix.registers import SrcRegister
from tt_sim.pe.tensix.util import DataFormatConversions
from tt_sim.perf.model import unit_cost_model
from tt_sim.util.bits import extract_bits, get_nth_bit
from tt_sim.util.conversion import conv_to_float, conv_to_uint32

#: Environment variable that turns the Src dvalid release check off, following
#: the ``TT_SIM_DISABLE_ALIGNMENT_CHECKS`` convention in
#: ``tt_sim/network/alignment.py``.
DISABLE_DVALID_CHECK_ENV_VAR = "TT_SIM_DISABLE_DVALID_CHECKS"

_TRUTHY = {"1", "true", "yes", "on"}

#: Read once at import: this is consulted on every math op that clears dvalid.
#: Call :func:`set_dvalid_checking_enabled` after mutating the environment.
_dvalid_checks_disabled = (
    os.environ.get(DISABLE_DVALID_CHECK_ENV_VAR, "").strip().lower() in _TRUTHY
)


class SrcDvalidError(RuntimeError):
    """A Matrix Unit instruction released a Src bank the Matrix Unit did not own.

    Clearing dvalid is the *release* half of the two-bank Src ownership
    handshake: the unpackers hand a bank over by setting
    ``AllowedClient = MatrixUnit`` (an ``UNPACR`` with ``SetDvalid``, an
    ``UNPACR_NOP`` carrying ``set_dvalid``, or a ``SETDVALID``), and the FPU
    hands it back by clearing it. Releasing a bank that is already owned by the
    unpackers is ``NonContractualBehavior``: the vendor reference simulator
    fails it outright (``ttsim`` ``src/tensix.cpp``, ``math_clear_src_valid``
    and ``TENSIX_EXECUTE_CLEARDVALID``, ``TTSIM_VERIFY(p_tensix->src_a_valid &
    (1 << p_tensix->src_a_matrix_bank), NonContractualBehavior, "SrcA bank is
    not valid")``), and on silicon it desynchronises the two banks' pointers so
    that a later acquire waits for a release that never comes.

    The instructions that read Src wait for ownership at the Wait Gate, so they
    cannot trip this. The ones that can are exactly those that clear dvalid
    *without* reading Src -- ``CLEARDVALID`` and ``SETRWC``'s ``clear_ab_vld``
    -- which is the shape a kernel reaches for when a math op did not carry
    ``clear_dvalid`` inline. That is a real, and easy, software mistake: see
    ``docs/plans/matrix-unit-thread-contention.md``, where an unmatched acquire
    wedged a Blackhole card until it was power-cycled.
    """


def set_dvalid_checking_enabled(enabled: bool) -> None:
    """Force the release check on or off, ignoring the environment (for tests)."""
    global _dvalid_checks_disabled
    _dvalid_checks_disabled = not enabled


class MatrixUnit(TensixBackendUnit):
    """
    Performs operations on srcA and srcB, writing results to dst register. Most obvious
    is matrix multiplication, but other element wise operations supported too. This implementation
    performs all real number calculations in FP32, converting from the data format in srcA and srcB
    (e.g. BF16, TF32, FP16) into FP32, and then the result is converted back to the data format
    and then written to dst. This will give slightly different results than on the Tenstorrent
    hardware.

    Based on description and code snippets at
    https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/MatrixUnit.md
    """

    OPCODE_TO_HANDLER = {
        "ZEROACC": "handle_zeroacc",
        "SETRWC": "handle_setrwc",
        "ELWADD": "handle_elwadd",
        "ELWSUB": "handle_elwsub",
        "ELWMUL": "handle_elwmul",
        "ZEROSRC": "handle_zerosrc",
        "INCRWC": "handle_incrwc",
        "CLEARDVALID": "handle_cleardvalid",
        "GATESRCRST": "handle_gatesrcrst",
        "TRNSPSRCB": "handle_trnspsrcb",
        "MOVB2A": "handle_movb2a",
        "MOVA2D": "handle_mova2d",
        "MOVB2D": "handle_movb2d",
        "MOVD2A": "handle_movd2a",
        "MOVD2B": "handle_movd2b",
        "MVMUL": "handle_mvmul",
        "DOTPV": "handle_dotpv",
        "GAPOOL": "handle_gapool",
        "GMPOOL": "handle_gmpool",
    }

    def __init__(self, backend):
        self.srcABank = 0
        self.srcBBank = 0
        super().__init__(backend, MatrixUnit.OPCODE_TO_HANDLER, "Matrix")
        # Phase 5 of docs/plans/event-driven-pump.md: the matrix unit is the
        # first (and so far only) consumer of the cycle-cost tables. ``None``
        # unless TT_SIM_COST_MODEL is set, in which case every op retires in
        # the tick it was issued exactly as before.
        self.cost_model = unit_cost_model(
            "MATH", "blackhole" if backend.blackhole else "wormhole"
        )

    def instruction_occupancy(self, instruction_name, issue_thread):
        """Cycles this op occupies the FPU, from ``tensix_instruction_costs.yaml``.

        Only reached when ``TT_SIM_COST_MODEL`` is set (:attr:`cost_model` is
        ``None`` otherwise). Two paths:

        * the ops the table marks ``scales_with: fidelity_phases`` (``MVMUL``,
          ``DOTPV``, ``GAPOOL``, ``ELWMUL``) are costed as a function of the
          fidelity phase this instruction runs at — the whole point of Phase 5
          for this unit, and the one cost in the MATH table that a flat number
          cannot express. See
          :meth:`~tt_sim.perf.model.UnitCostModel.fidelity_occupancy`;
        * everything else takes the table's occupancy, which for this unit is
          ``1 / throughput_ipc`` (``isa_doc_derived``).

        The phase is read *before* the handler runs, because ``ADDR_MOD`` may
        advance ``RWC.FidelityPhase`` on the way out and the cost belongs to
        the phase the instruction actually executed at.
        """
        model = self.cost_model
        if model is None:
            return None
        if model.scales_with_fidelity(instruction_name):
            phase = self.determine_fidelity_phase(
                issue_thread, self.backend.getRWC(issue_thread)
            )
            cycles = model.fidelity_occupancy(phase)
            if cycles is not None:
                return cycles
        return model.occupancy(instruction_name)

    def getSrcA(self):
        return self.backend.getSrcA(self.srcABank)

    def getSrcB(self):
        return self.backend.getSrcB(self.srcBBank)

    def _read_addr_mode(self, instruction_info, instr_args):
        # The math ``addr_mode`` field is 3 bits on Blackhole (raw bits 16:14)
        # but only 2 bits on Wormhole (16:15) — Blackhole has 8 ADDR_MOD
        # sections, Wormhole 4. The shared instruction table encodes the
        # Wormhole start_bit (15), so on Blackhole its low bit (14) is dropped
        # and the value is shifted: ADDR_MOD_1/2/4/5 silently decode as
        # 0/1/2/2, losing the matmul MOP's dest-reset mods (ADDR_MOD_5) so the
        # second K-face never accumulates. Read the true field from the raw
        # word on Blackhole. See docs/plans/blackhole-support.md (`six`).
        if self.backend.blackhole:
            return extract_bits(instruction_info["raw_instruction"], 3, 14)
        return instr_args["addr_mode"]

    def _read_dst_field(self, instruction_info, instr_args):
        # ``dst`` (the instruction's own Dst row offset) runs 14:0 in the shared
        # instruction table, because the next field along -- Wormhole's
        # ``addr_mode`` -- starts at bit 15. Blackhole widened ``addr_mode``
        # downwards into bit 14 (see ``_read_addr_mode``), so on Blackhole that
        # bit belongs to the addressing mode and reading it as part of ``dst``
        # adds a spurious 16384 whenever an odd ADDR_MOD is selected. Read the
        # true 14-bit field from the raw word instead.
        if self.backend.blackhole:
            return extract_bits(instruction_info["raw_instruction"], 14, 0)
        return instr_args["dst"]

    def _read_pool_addr_mode(self, instruction_info, instr_args):
        # GAPOOL and GMPOOL keep their addressing-mode field at bit 15 on
        # Blackhole (where it is renamed ``pool_addr_mode``), but widen it to 3
        # bits: raw 17:15 versus Wormhole's 16:15. That is a different position
        # to every other math instruction (which move to 16:14, see
        # ``_read_addr_mode``), so these two need their own reader — using the
        # generic one would pick up bit 14, which here is ``max_pool_index_en``.
        # See docs/plans/blackhole-support.md (Phase 8).
        if self.backend.blackhole:
            return extract_bits(instruction_info["raw_instruction"], 3, 15)
        return instr_args["addr_mode"]

    def _read_movb2d_instr_mod(self, instruction_info, instr_args):
        # MOVB2D's instruction modifier sits at raw 13:11 on Blackhole (where
        # it is renamed ``movb2d_instr_mod``) but at 14:12 on Wormhole, because
        # Blackhole widened ``addr_mode`` downwards into bit 14. The shared
        # instruction table encodes the Wormhole position, so on Blackhole
        # every modifier would be read doubled (and the Move4Rows bit lost into
        # addr_mode's territory). Read the true field from the raw word.
        if self.backend.blackhole:
            return extract_bits(instruction_info["raw_instruction"], 3, 11)
        return instr_args["instr_mod"]

    def handle_cleardvalid(self, instruction_info, issue_thread, instr_args):
        reset = instr_args["reset"] & 0x1
        keepReadingSameSrc = (instr_args["reset"] >> 1) & 0x1
        flipSrcA = instr_args["cleardvalid"] & 0x1
        flipSrcB = (instr_args["cleardvalid"] >> 1) & 0x1

        if reset:
            self.srcABank = 0
            self.srcBBank = 0
            self.backend.unpacker_units[0].srcBank = 0
            self.backend.unpacker_units[1].srcBank = 0
            self.backend.getSrcA(0).allowedClient = SrcRegister.SrcClient.Unpackers
            self.backend.getSrcA(1).allowedClient = SrcRegister.SrcClient.Unpackers
            self.backend.getSrcB(0).allowedClient = SrcRegister.SrcClient.Unpackers
            self.backend.getSrcB(1).allowedClient = SrcRegister.SrcClient.Unpackers
        else:
            # CLEARDVALID's release is not gated on CLR_DVALID_Src{A,B}_Disable
            # (ttsim's TENSIX_EXECUTE_CLEARDVALID clears unconditionally, unlike
            # math_clear_src_valid), so this cannot go through
            # optionally_flip_src_banks -- but the ownership check is the same
            # one, and this is the instruction most likely to trip it.
            if flipSrcA:
                self._release_src_bank(
                    "SrcA", self.srcABank, issue_thread, "CLEARDVALID"
                )
                if not keepReadingSameSrc:
                    self.srcABank ^= 1
            if flipSrcB:
                self._release_src_bank(
                    "SrcB", self.srcBBank, issue_thread, "CLEARDVALID"
                )
                if not keepReadingSameSrc:
                    self.srcBBank ^= 1

    def handle_gatesrcrst(self, instruction_info, issue_thread, instr_args):
        # GATESRCRST forcibly invalidates the one-slot operand cache sitting
        # between SrcB and the Matrix Unit and resets the SrcA/SrcB pipeline
        # gating to "don't gate". This functional model reads SrcA/SrcB
        # directly on every op and models neither the operand cache nor the
        # gating mechanism, so there is no state to reset here; it is a no-op.
        pass

    def handle_trnspsrcb(self, instruction_info, issue_thread, instr_args):
        # Transpose the 16x16 matrix held in SrcB rows [16, 32) in place. The
        # ISA wait-gate condition (the FPU must own the SrcB bank before
        # dispatch, as in ttsim's TENSIX_EXECUTE_TRNSPSRCB) is enforced in
        # WaitGate.checkIfFPUInstructionShouldStall before the instruction
        # reaches this unit -- it was not, for years, because the gate's table
        # spelled the opcode TRANSPSRCB -- so only the swap is performed here.
        srcB = self.getSrcB()
        rowBase = 16
        for i in range(16):
            for j in range(i):
                ij = srcB[rowBase + i, j]
                ji = srcB[rowBase + j, i]
                srcB[rowBase + i, j] = ji
                srcB[rowBase + j, i] = ij

    def handle_movb2a(self, instruction_info, issue_thread, instr_args):
        srcBRow = instr_args["srcb"]
        move4Rows = (instr_args["instr_mod"] >> 1) & 0x1
        addrMod = self._read_addr_mode(instruction_info, instr_args)
        srcARow = instr_args["srca"]

        stateID = self.backend.getThreadConfigValue(
            issue_thread, "CFG_STATE_ID_StateID"
        )
        flushDenormals = not self.getConfigValue(
            stateID, "ALU_ACC_CTRL_Zero_Flag_disabled_src"
        )

        rwc = self.backend.getRWC(issue_thread)

        # Determine the row range
        srcARow += rwc.SrcA
        srcBRow += rwc.SrcB
        if move4Rows:
            numRows = 4
            srcARow &= 0x3C
            srcBRow &= 0x3C
        else:
            numRows = 1
            srcARow &= 0x3F
            srcBRow &= 0x3F

        # Now copy the row(s) from SrcB into SrcA
        srcA = self.getSrcA()
        srcB = self.getSrcB()
        for i in range(numRows):
            for j in range(16):
                if get_nth_bit(
                    self.backend.vector_unit.laneConfigValue(
                        int(j / 2), VectorUnit.BLOCK_DEST_MOV
                    ),
                    j & 1,
                ):
                    continue
                srcVal = srcB[srcBRow, j]
                if flushDenormals and not (srcVal & 0xFF):
                    srcVal = 0
                srcA[srcARow, j] = srcVal
            srcARow += 1
            srcBRow += 1

        # Advance the RWCs
        rwc.applyAddrMod(issue_thread, addrMod)

    def handle_mova2d(self, instruction_info, issue_thread, instr_args):
        dstRow = instr_args["dst"]
        move8Rows = (instr_args["instr_mod"] >> 1) & 0x1
        addrMod = self._read_addr_mode(instruction_info, instr_args)
        srcRow = instr_args["src"]
        useDst32bLo = instr_args["dest_32b_lo"]

        stateID = self.backend.getThreadConfigValue(
            issue_thread, "CFG_STATE_ID_StateID"
        )

        if self.getConfigValue(stateID, "ALU_FORMAT_SPEC_REG_SrcA_override"):
            srcAFmt = self.getConfigValue(stateID, "ALU_FORMAT_SPEC_REG_SrcA_val")
        else:
            srcAFmt = self.getConfigValue(stateID, "ALU_FORMAT_SPEC_REG0_SrcA")

        if self.getThreadConfigValue(issue_thread, "FP16A_FORCE_Enable"):
            use8bExponent = False
        elif srcAFmt in [
            DataFormat.FP32,
            DataFormat.TF32,
            DataFormat.BF16,
            DataFormat.BFP8,
            DataFormat.BFP4,
            DataFormat.BFP2,
            DataFormat.INT32,
            DataFormat.UINT16,
        ]:
            use8bExponent = True
        else:
            use8bExponent = False

        flushDenormals = not self.getConfigValue(
            stateID, "ALU_ACC_CTRL_Zero_Flag_disabled_src"
        )
        rwc = self.backend.getRWC(issue_thread)

        # Determine the row range
        dstRow += self.getThreadConfigValue(
            issue_thread, "DEST_TARGET_REG_CFG_MATH_Offset"
        )
        dstRow += rwc.Dst + self.getConfigValue(stateID, "DEST_REGW_BASE_Base")
        srcRow += rwc.SrcA

        if move8Rows:
            numRows = 8
            dstRow &= 0x3F8
            srcRow &= 0x38
        else:
            numRows = 1
            dstRow &= 0x3FF
            srcRow &= 0x3F

        # Now copy the row(s)
        for i in range(numRows):
            for j in range(16):
                if get_nth_bit(
                    self.backend.vector_unit.laneConfigValue(
                        int(j / 2), VectorUnit.BLOCK_DEST_MOV
                    ),
                    j & 1,
                ):
                    continue
                srcAVal = self.backend.getSrcA(self.srcABank)[srcRow, j]
                if flushDenormals and not (srcAVal & 0xFF):
                    srcAVal = 0
                val16b = (
                    DataFormatConversions.removeLowMantissa(srcAVal)
                    if use8bExponent
                    else DataFormatConversions.removeHighExponent(srcAVal)
                )
                if srcAFmt == DataFormat.TF32:
                    lowMantissa = ((srcAVal >> 8) & 7) << 13
                    if useDst32bLo:
                        lowMantissa |= val16b
                    # dst holds TF32 as sign,himan(7b),exp(8b),loman(3b),zeros(13b)
                    self.getDst().setDst32b(dstRow, j, (val16b << 16) | lowMantissa)
                elif useDst32bLo:
                    self.getDst().setDst32b(
                        dstRow,
                        j,
                        (self.getDst().getDst32b(dstRow, j) & 0xFFFF0000) | val16b,
                    )
                else:
                    self.getDst().setDst16b(dstRow, j, val16b)
            dstRow += 1
            srcRow += 1

        # Advance the RWCs
        rwc.applyAddrMod(issue_thread, addrMod)

    def handle_movb2d(self, instruction_info, issue_thread, instr_args):
        dstRow = instr_args["dst"]
        instrMod = self._read_movb2d_instr_mod(instruction_info, instr_args)
        broadcastCol0 = get_nth_bit(instrMod, 0)
        broadcast1RowTo8 = get_nth_bit(instrMod, 1)
        move4Rows = get_nth_bit(instrMod, 2)
        addrMod = self._read_addr_mode(instruction_info, instr_args)
        srcRow = instr_args["src"]
        useDst32bLo = instr_args["dest_32b_lo"]

        stateID = self.backend.getThreadConfigValue(
            issue_thread, "CFG_STATE_ID_StateID"
        )

        # Determine the data formats. Note that (as with MOVA2D / MOVD2B)
        # SrcAFmt is deliberately used to pick the Dst style; this is not a typo.
        if self.getConfigValue(stateID, "ALU_FORMAT_SPEC_REG_SrcA_override"):
            srcAFmt = self.getConfigValue(stateID, "ALU_FORMAT_SPEC_REG_SrcA_val")
        else:
            srcAFmt = self.getConfigValue(stateID, "ALU_FORMAT_SPEC_REG0_SrcA")
        if self.getConfigValue(stateID, "ALU_FORMAT_SPEC_REG_SrcB_override"):
            srcBFmt = self.getConfigValue(stateID, "ALU_FORMAT_SPEC_REG_SrcB_val")
        else:
            srcBFmt = self.getConfigValue(stateID, "ALU_FORMAT_SPEC_REG1_SrcB")
        # On Blackhole both selects are implied from the format SrcB was last
        # written in, each behind its own disable bit.
        srcAFmt = self.implied_srcB_format(
            issue_thread, srcAFmt, "DISABLE_IMPLIED_SRCA_FMT_Base"
        )
        srcBFmt = self.implied_srcB_format(issue_thread, srcBFmt)

        if self.getThreadConfigValue(issue_thread, "FP16A_FORCE_Enable"):
            use8bExponent = False
        elif srcAFmt in [
            DataFormat.FP32,
            DataFormat.TF32,
            DataFormat.BF16,
            DataFormat.BFP8,
            DataFormat.BFP4,
            DataFormat.BFP2,
            DataFormat.INT32,
            DataFormat.UINT16,
        ]:
            use8bExponent = True
        else:
            use8bExponent = False

        flushDenormals = not self.getConfigValue(
            stateID, "ALU_ACC_CTRL_Zero_Flag_disabled_src"
        )
        rwc = self.backend.getRWC(issue_thread)

        # Determine the row range
        dstRow += self.getThreadConfigValue(
            issue_thread, "DEST_TARGET_REG_CFG_MATH_Offset"
        )
        dstRow += rwc.Dst + self.getConfigValue(stateID, "DEST_REGW_BASE_Base")
        srcRow += rwc.SrcB

        if broadcast1RowTo8:
            numRows = 8
            dstRow &= 0x3F8
            srcRow &= 0x3F
        elif move4Rows:
            numRows = 4
            dstRow &= 0x3FC
            srcRow &= 0x3C
        else:
            numRows = 1
            dstRow &= 0x3FF
            srcRow &= 0x3F

        # Now copy the row(s) from SrcB into Dst
        srcB = self.getSrcB()
        for _ in range(numRows):
            for j in range(16):
                if get_nth_bit(
                    self.backend.vector_unit.laneConfigValue(
                        int(j / 2), VectorUnit.BLOCK_DEST_MOV
                    ),
                    j & 1,
                ):
                    continue

                srcBVal = srcB[srcRow, 0 if broadcastCol0 else j]
                if flushDenormals and not (srcBVal & 0xFF):
                    srcBVal = 0

                # Src holds a datum as Sign,Man(10b),Exp(8b); narrow it to the
                # precision the SrcB format actually carries.
                if srcBFmt == DataFormat.FP16:
                    srcBVal &= 0x7FF1F  # drop the high 3 exponent bits
                elif srcBFmt != DataFormat.TF32:
                    srcBVal &= 0x7F8FF  # drop the low 3 mantissa bits

                val16b = (
                    DataFormatConversions.removeLowMantissa(srcBVal)
                    if use8bExponent
                    else DataFormatConversions.removeHighExponent(srcBVal)
                )

                if srcAFmt == DataFormat.TF32:
                    if useDst32bLo:
                        # Undefined behaviour: on Blackhole the write data is
                        # corrupted (HW erratum TEN-4245) and the Wormhole
                        # behaviour is unlikely to be useful. ttsim declines it
                        # too, so there is nothing to model against.
                        raise NotImplementedError(
                            "MOVB2D with SrcAFmt == TF32 and UseDst32bLo is undefined behaviour"
                        )
                    lowMantissa = ((srcBVal >> 8) & 7) << 13
                    # dst holds TF32 as sign,himan(7b),exp(8b),loman(3b),zeros(13b)
                    self.getDst().setDst32b(dstRow, j, (val16b << 16) | lowMantissa)
                elif useDst32bLo:
                    # Only useful if software has deliberately packed two
                    # bf16/fp16 values into 32 bits and written them to Dst32b
                    self.getDst().setDst32b(
                        dstRow,
                        j,
                        (self.getDst().getDst32b(dstRow, j) & 0xFFFF0000) | val16b,
                    )
                else:
                    self.getDst().setDst16b(dstRow, j, val16b)
            dstRow += 1
            if not broadcast1RowTo8:
                srcRow += 1

        # Advance the RWCs
        rwc.applyAddrMod(issue_thread, addrMod)

    def handle_movd2a(self, instruction_info, issue_thread, instr_args):
        dstRow = instr_args["dst"]
        move4Rows = (instr_args["instr_mod"] >> 1) & 0x1
        addrMod = self._read_addr_mode(instruction_info, instr_args)
        srcRow = instr_args["src"]
        useDst32bLo = instr_args["dest_32b_lo"]

        stateID = self.backend.getThreadConfigValue(
            issue_thread, "CFG_STATE_ID_StateID"
        )

        # Determine the data formats
        if self.getThreadConfigValue(issue_thread, "FP16A_FORCE_Enable"):
            useDst32b = False
            srcAStyle = DataFormat.FP16
        else:
            if self.getConfigValue(stateID, "ALU_FORMAT_SPEC_REG_SrcA_override"):
                srcAFmt = self.getConfigValue(stateID, "ALU_FORMAT_SPEC_REG_SrcA_val")
            else:
                srcAFmt = self.getConfigValue(stateID, "ALU_FORMAT_SPEC_REG0_SrcA")
            srcAFmt = self.implied_srcA_format(issue_thread, srcAFmt)

            useDst32b = self.getConfigValue(
                stateID, "ALU_ACC_CTRL_Fp32_enabled"
            ) or self.getConfigValue(stateID, "ALU_ACC_CTRL_INT8_math_enabled")

            if srcAFmt in [
                DataFormat.FP32,
                DataFormat.BF16,
                DataFormat.BFP8,
                DataFormat.BFP4,
                DataFormat.BFP2,
                DataFormat.INT32,
                DataFormat.UINT16,
            ]:
                srcAStyle = DataFormat.BF16
            elif srcAFmt in [
                DataFormat.FP16,
                DataFormat.BFP8_b,
                DataFormat.BFP4_b,
                DataFormat.BFP2_b,
                DataFormat.INT8,
            ]:
                srcAStyle = DataFormat.FP16
            else:
                # SrcAFmt == TF32
                srcAStyle = DataFormat.TF32

        rwc = self.backend.getRWC(issue_thread)

        # Determine the row range
        dstRow += self.getThreadConfigValue(
            issue_thread, "DEST_TARGET_REG_CFG_MATH_Offset"
        )
        dstRow += rwc.Dst + self.getConfigValue(stateID, "DEST_REGW_BASE_Base")
        srcRow += rwc.SrcA

        if move4Rows:
            numRows = 4
            dstRow &= 0x3FC
            srcRow &= 0x3C
        else:
            numRows = 1
            dstRow &= 0x3FF
            srcRow &= 0x3F

        # Now copy the row(s) from Dst into SrcA
        srcA = self.getSrcA()
        for i in range(numRows):
            for j in range(16):
                if get_nth_bit(
                    self.backend.vector_unit.laneConfigValue(
                        int(j / 2), VectorUnit.BLOCK_DEST_MOV
                    ),
                    j & 1,
                ):
                    continue

                if useDst32b:
                    # Read from Dst in 32-bit mode
                    dstVal = self.getDst().getDst32b(dstRow, j)
                    if useDst32bLo:
                        # Only useful if software has deliberately packed two
                        # bf16/fp16 values into 32 bits and written them to Dst32b
                        dstVal = (dstVal << 16) | (dstVal & 0xFFFF)

                    if srcAStyle == DataFormat.BF16:
                        # Treat dstVal as fp32 or tf32, truncate to bf16
                        srcAVal = DataFormatConversions.ShuffleBF16(dstVal >> 16)
                    elif srcAStyle == DataFormat.FP16:
                        srcAVal = DataFormatConversions.ShuffleFP16(dstVal >> 16)
                    elif not useDst32bLo:
                        # Treat dstVal as fp32 or tf32, truncate to tf32
                        srcAVal = DataFormatConversions.ShuffleTF32(dstVal >> 13)
                    else:
                        # The 13 bits discarded by the fp32 -> tf32 conversion
                        srcAVal = dstVal & 0x1FFF
                else:
                    # Read from Dst in 16-bit mode. Both of the combinations
                    # rejected here are documented as undefined behaviour and
                    # ttsim declines them as well, so there is no reference to
                    # model them against.
                    if useDst32bLo:
                        raise NotImplementedError(
                            "MOVD2A with UseDst32bLo and a 16-bit Dst is undefined behaviour"
                        )
                    dstVal = self.getDst().getDst16b(dstRow, j)
                    if srcAStyle == DataFormat.BF16:
                        # Treat dstVal as bf16
                        srcAVal = DataFormatConversions.ShuffleBF16(dstVal)
                    elif srcAStyle == DataFormat.FP16:
                        # Treat dstVal as fp16 (int8 is overlaid onto fp16 here)
                        srcAVal = DataFormatConversions.ShuffleFP16(dstVal)
                    else:
                        raise NotImplementedError(
                            "MOVD2A with SrcAStyle == TF32 and a 16-bit Dst is undefined behaviour"
                        )

                srcA[srcRow, j] = srcAVal
            dstRow += 1
            srcRow += 1

        # Advance the RWCs
        rwc.applyAddrMod(issue_thread, addrMod)

    def handle_movd2b(self, instruction_info, issue_thread, instr_args):
        dstRow = instr_args["dst"]
        move4Rows = (instr_args["instr_mod"] >> 1) & 0x1
        addrMod = self._read_addr_mode(instruction_info, instr_args)
        srcRow = instr_args["src"]
        useDst32bLo = instr_args["dest_32b_lo"]

        stateID = self.backend.getThreadConfigValue(
            issue_thread, "CFG_STATE_ID_StateID"
        )

        # Determine the data formats. Note that (as with MOVA2D) SrcAFmt is
        # deliberately used to select the SrcB style; this is not a typo.
        if self.getThreadConfigValue(issue_thread, "FP16A_FORCE_Enable"):
            useDst32b = False
            srcBStyle = DataFormat.FP16
        else:
            if self.getConfigValue(stateID, "ALU_FORMAT_SPEC_REG_SrcA_override"):
                srcAFmt = self.getConfigValue(stateID, "ALU_FORMAT_SPEC_REG_SrcA_val")
            else:
                srcAFmt = self.getConfigValue(stateID, "ALU_FORMAT_SPEC_REG0_SrcA")

            useDst32b = self.getConfigValue(
                stateID, "ALU_ACC_CTRL_Fp32_enabled"
            ) or self.getConfigValue(stateID, "ALU_ACC_CTRL_INT8_math_enabled")

            if srcAFmt in [
                DataFormat.FP32,
                DataFormat.BF16,
                DataFormat.BFP8,
                DataFormat.BFP4,
                DataFormat.BFP2,
                DataFormat.INT32,
                DataFormat.UINT16,
            ]:
                srcBStyle = DataFormat.BF16
            elif srcAFmt in [
                DataFormat.FP16,
                DataFormat.BFP8_b,
                DataFormat.BFP4_b,
                DataFormat.BFP2_b,
                DataFormat.INT8,
            ]:
                srcBStyle = DataFormat.FP16
            else:
                # SrcAFmt == TF32
                srcBStyle = DataFormat.TF32

        rwc = self.backend.getRWC(issue_thread)

        # Determine the row range
        dstRow += self.getThreadConfigValue(
            issue_thread, "DEST_TARGET_REG_CFG_MATH_Offset"
        )
        dstRow += rwc.Dst + self.getConfigValue(stateID, "DEST_REGW_BASE_Base")
        srcRow += rwc.SrcB

        if move4Rows:
            numRows = 4
            dstRow &= 0x3FC
            srcRow &= 0x3C
        else:
            numRows = 1
            dstRow &= 0x3FF
            srcRow &= 0x3F

        # Now copy the row(s) from Dst into SrcB
        srcB = self.getSrcB()
        for i in range(numRows):
            for j in range(16):
                if get_nth_bit(
                    self.backend.vector_unit.laneConfigValue(
                        int(j / 2), VectorUnit.BLOCK_DEST_MOV
                    ),
                    j & 1,
                ):
                    continue

                if useDst32b:
                    # Read from Dst in 32-bit mode
                    dstVal = self.getDst().getDst32b(dstRow, j)
                    if useDst32bLo:
                        # Only useful if software has deliberately packed two
                        # bf16/fp16 values into 32 bits and written them to Dst32b
                        dstVal = (dstVal << 16) | (dstVal & 0xFFFF)

                    if srcBStyle == DataFormat.BF16:
                        # Treat dstVal as fp32 or tf32, truncate to bf16
                        srcBVal = DataFormatConversions.ShuffleBF16(dstVal >> 16)
                    elif srcBStyle == DataFormat.FP16:
                        srcBVal = DataFormatConversions.ShuffleFP16(dstVal >> 16)
                    elif not useDst32bLo:
                        # Treat dstVal as fp32 or tf32, truncate to tf32
                        srcBVal = DataFormatConversions.ShuffleTF32(dstVal >> 13)
                    else:
                        # The 13 bits discarded by the fp32 -> tf32 conversion
                        srcBVal = dstVal & 0x1FFF
                else:
                    # Read from Dst in 16-bit mode
                    dstVal = self.getDst().getDst16b(dstRow, j)
                    if srcBStyle == DataFormat.BF16:
                        # Treat dstVal as bf16
                        srcBVal = DataFormatConversions.ShuffleBF16(dstVal)
                    elif srcBStyle == DataFormat.FP16:
                        # Treat dstVal as fp16 (int8 is overlaid onto fp16 here)
                        srcBVal = DataFormatConversions.ShuffleFP16(dstVal)
                    else:
                        # dstVal isn't wide enough to hold fp32/tf32 data; the ISA
                        # documents this combination as undefined behaviour
                        srcBVal = 0

                srcB[srcRow, j] = srcBVal
            dstRow += 1
            srcRow += 1

        # Advance the RWCs
        rwc.applyAddrMod(issue_thread, addrMod)

    def handle_incrwc(self, instruction_info, issue_thread, instr_args):
        srcAInc = instr_args["rwc_a"]
        srcBInc = instr_args["rwc_b"]
        dstInc = instr_args["rwc_d"]
        rwc_CR = instr_args["rwc_cr"]
        srcACr = get_nth_bit(rwc_CR, 0)
        srcBCr = get_nth_bit(rwc_CR, 1)
        dstCr = get_nth_bit(rwc_CR, 2)

        rwc = self.backend.getRWC(issue_thread)

        if self.getDiagnosticSettings().reportFPUCalculations():
            print(
                f"FPU: incrwc AInc={srcAInc} BInc={srcBInc} dstInc={dstInc} by thread {issue_thread}"
            )

        if srcACr:
            rwc.SrcA_Cr += srcAInc
            rwc.SrcA = rwc.SrcA_Cr
        else:
            rwc.SrcA += srcAInc

        if srcBCr:
            rwc.SrcB_Cr += srcBInc
            rwc.SrcB = rwc.SrcB_Cr
        else:
            rwc.SrcB += srcBInc

        if dstCr:
            rwc.Dst_Cr += dstInc
            rwc.Dst = rwc.Dst_Cr
        else:
            rwc.Dst += dstInc

    def handle_zerosrc(self, instruction_info, issue_thread, instr_args):
        clearSrcABank = [False] * 2
        clearSrcBBank = [False] * 2

        clearSrcA = get_nth_bit(instr_args["src_mask"], 0)
        clearSrcB = get_nth_bit(instr_args["src_mask"], 1)
        bothBanks = instr_args["bank_mask"]
        singleBankMatrixUnit = instr_args["write_mode"]
        negativeInfSrcA = instr_args["zero_val"]

        if self.getDiagnosticSettings().reportFPUCalculations():
            print(
                f"FPU: perform zerosrc for srcA={clearSrcA}, srcB={clearSrcB} by thread {issue_thread}"
            )

        if clearSrcA:
            if bothBanks:
                clearSrcABank[0] = True
                clearSrcABank[1] = True
            elif singleBankMatrixUnit:
                clearSrcABank[self.srcABank] = True
            else:
                clearSrcABank[self.backend.unpacker_units[0].srcBank] = True

        if clearSrcB:
            if bothBanks:
                clearSrcBBank[0] = True
                clearSrcBBank[1] = True
            elif singleBankMatrixUnit:
                clearSrcBBank[self.srcBBank] = True
            else:
                clearSrcBBank[self.backend.unpacker_units[1].srcBank] = True

        # Do the clearing
        for bank in range(2):
            for i in range(64):
                for j in range(16):
                    if clearSrcABank[bank]:
                        self.backend.getSrcA(bank)[i, j] = ~0 if negativeInfSrcA else 0
                    if clearSrcBBank[bank]:
                        self.backend.getSrcB(bank)[i, j] = 0

    def handle_gapool(self, instruction_info, issue_thread, instr_args):
        dstRow = self._read_dst_field(instruction_info, instr_args)
        addrMod = self._read_pool_addr_mode(instruction_info, instr_args)
        flipSrcA = instr_args["clear_dvalid"] & 0x1
        flipSrcB = instr_args["clear_dvalid"] & 0x2

        self.perform_mvmul(
            issue_thread, dstRow, addrMod, False, flipSrcA, flipSrcB, 4, "GAPOOL"
        )

    @staticmethod
    def _gmpool_read_dst(dstVal, dstStyle):
        """Decode a Dst datum into GMPOOL's (sign, exp(9b), magOrMan(10b)) form."""
        sign = (dstVal >> 31) & 1
        if dstStyle == DataFormat.FP16:
            # High 16 bits of Dst hold Sign,Man(10b),Exp(5b)
            return sign, ((dstVal >> 16) & 0x1F) + 15, (dstVal >> 21) & 0x3FF
        # BF16 / TF32: high 16 bits of Dst hold Sign,Man(7b),Exp(8b)
        exponent = ((dstVal >> 16) & 0xFF) + 127
        magOrMan = ((dstVal >> 24) & 0x7F) << 3
        if dstStyle == DataFormat.TF32:
            # The next three bits of Dst hold more of the mantissa
            magOrMan += (dstVal >> 13) & 7
        return sign, exponent, magOrMan

    @staticmethod
    def _gmpool_write_dst(datum, dstStyle):
        """Re-encode a GMPOOL datum for Dst (the inverse of _gmpool_read_dst)."""
        sign, exponent, magOrMan = datum
        if dstStyle == DataFormat.FP16:
            if (exponent & 0x3F) == 0:
                return 0
            # Can wrap around if the SrcB scaler was not 1.0
            exponent -= 15
            return (sign << 31) | (magOrMan << 21) | ((exponent & 0x1F) << 16)
        if exponent == 0:
            return 0
        # Can wrap around if the SrcB scaler was not 1.0
        exponent -= 127
        x = (sign << 31) | ((magOrMan & 0x3F8) << 21) | ((exponent & 0xFF) << 16)
        if dstStyle == DataFormat.TF32:
            x += (magOrMan & 7) << 13
        return x

    @staticmethod
    def _gmpool_read_and_scale_src(srcAVal, srcAStyle, srcBVal):
        """Read a SrcA datum, scaled by the exponent of the paired SrcB datum.

        Src holds BF16 as Sign,Man(10b),Exp(8b) with the bottom three mantissa
        bits unused, TF32 as Sign,Man(10b),Exp(8b) and FP16 as
        Sign,Man(10b),Zero(3b),Exp(5b).
        """
        srcAExponent = srcAVal & 0xFF
        srcBExponent = srcBVal & 0xFF
        if srcBExponent == 0:
            # Scaling by zero, which is effectively minus infinity
            return 1, 0x1FF, 0x3FF
        if srcAExponent == 0:
            # Flush denormals to zero
            return 0, 0, 0
        sign = (srcAVal >> 18) & 1
        magOrMan = (srcAVal >> 8) & 0x3FF
        if srcAStyle == DataFormat.FP16:
            return sign, (srcAExponent & 0x1F) + (srcBExponent & 0x1F), magOrMan
        if srcAStyle == DataFormat.TF32:
            return sign, srcAExponent + srcBExponent, magOrMan
        # BF16, whose bottom three mantissa bits are not carried
        return sign, srcAExponent + srcBExponent, magOrMan & 0x3F8

    @staticmethod
    def _gmpool_as_comparable(datum):
        sign, exponent, magOrMan = datum
        magnitude = (exponent << 10) + magOrMan
        return -magnitude if sign else magnitude

    def handle_gmpool(self, instruction_info, issue_thread, instr_args):
        """Reduce a 16x16 block of SrcA to one row by ``max`` along each column.

        The result is element-wise ``max``-ed with the top row of an aligned 4x16
        block of Dst and written back there, zeroing the other three rows. Each
        SrcA column is first scaled by the exponent of the corresponding datum of
        one SrcB row (so software passes a row of 1.0 when it wants no scaling).

        GMPOOL is the one consumer that sees a flag-cleared (ZEROACC'd) Dst row
        as **minus infinity** rather than as zero — all bits set, which decodes
        to the most negative datum in the comparison order below. That is how
        software gets a plain reduction out of an accumulating instruction: it
        ZEROACCs the destination first, and without the substitution a reduction
        over an entirely negative face would saturate at +0. See
        ``DstRegister.dstRowValid``; ttsim's ``read_dst{16,32}b<is_gmpool>``.
        """
        dstRow = instr_args["dst"]
        addrMod = self._read_pool_addr_mode(instruction_info, instr_args)
        flipSrcA = get_nth_bit(instr_args["clear_dvalid"], 0)
        flipSrcB = get_nth_bit(instr_args["clear_dvalid"], 1)

        # Only the 16x16 primitive is documented (and it is the only one
        # tt-metal's reduce LLK emits); ttsim rejects everything else too.
        if instr_args["instr_mod19"] != 1:
            raise NotImplementedError(
                f"GMPOOL only implements the 16x16 primitive, not instr_mod19={instr_args['instr_mod19']}"
            )
        if instr_args["max_pool_index_en"]:
            # argmax returns a non-linearly transformed index instead of (or
            # alongside) the max; the ISA calls its support partial and ttsim
            # declines it, so there is no reference to model it against.
            raise NotImplementedError(
                "GMPOOL argmax (max_pool_index_en) is not modelled"
            )

        stateID = self.backend.getThreadConfigValue(
            issue_thread, "CFG_STATE_ID_StateID"
        )

        # Determine the data formats
        if self.getThreadConfigValue(issue_thread, "FP16A_FORCE_Enable"):
            srcAStyle = DataFormat.FP16
            dstStyle = DataFormat.FP16
            useDst32b = False
        elif self.getConfigValue(stateID, "ALU_ACC_CTRL_INT8_math_enabled"):
            raise NotImplementedError("GMPOOL with INT8 math enabled is not modelled")
        else:
            if self.getConfigValue(stateID, "ALU_FORMAT_SPEC_REG_SrcA_override"):
                srcAFmt = self.getConfigValue(stateID, "ALU_FORMAT_SPEC_REG_SrcA_val")
            else:
                srcAFmt = self.getConfigValue(stateID, "ALU_FORMAT_SPEC_REG0_SrcA")
            srcAFmt = self.implied_srcA_format(issue_thread, srcAFmt)

            useDst32b = self.getConfigValue(stateID, "ALU_ACC_CTRL_Fp32_enabled")
            if srcAFmt in [
                DataFormat.FP32,
                DataFormat.BF16,
                DataFormat.BFP8,
                DataFormat.BFP4,
                DataFormat.BFP2,
                DataFormat.INT32,
                DataFormat.UINT16,
            ]:
                srcAStyle = DataFormat.BF16
                dstStyle = DataFormat.TF32 if useDst32b else DataFormat.BF16
            elif srcAFmt in [
                DataFormat.FP16,
                DataFormat.BFP8_b,
                DataFormat.BFP4_b,
                DataFormat.BFP2_b,
                DataFormat.INT8,
            ]:
                srcAStyle = DataFormat.FP16
                dstStyle = DataFormat.TF32 if useDst32b else DataFormat.FP16
            else:
                # SrcAFmt == TF32
                srcAStyle = DataFormat.TF32
                dstStyle = DataFormat.TF32 if useDst32b else DataFormat.BF16

        rwc = self.getRWC(issue_thread)

        # Determine the row range
        srcARow = rwc.SrcA & 0x30
        srcBRow = rwc.SrcB & 0x38
        dstRow += self.getThreadConfigValue(
            issue_thread, "DEST_TARGET_REG_CFG_MATH_Offset"
        )
        dstRow += rwc.Dst + self.getConfigValue(stateID, "DEST_REGW_BASE_Base")
        dstRow &= 0x3FC

        if self.getDiagnosticSettings().reportFPUCalculations():
            print(
                f"FPU: perform gmpool, dst row {dstRow}, srcA starts at {srcARow} "
                f"and srcB at {srcBRow} by thread {issue_thread}"
            )

        srcA = self.getSrcA()
        srcB = self.getSrcB()
        dst = self.getDst()

        # Iterate over the SrcA columns
        for j in range(16):
            # isGmpool: a row whose zero flag is clear reads as all-ones here
            # (minus infinity), not as its zeroed data.
            if useDst32b:
                dstVal = dst.getDst32b(dstRow, j, isGmpool=True)
            else:
                dstVal = dst.getDst16b(dstRow, j, isGmpool=True) << 16
            maximum = self._gmpool_read_dst(dstVal, dstStyle)

            # Iterate over the SrcA rows (and the SrcB columns). The visitation
            # order is tweaked exactly as the hardware does, which only makes a
            # difference when resolving argmax ties.
            for i_ in range(16):
                i = (i_ ^ 4) if i_ < 8 else i_
                x = self._gmpool_read_and_scale_src(
                    srcA[srcARow + i, j], srcAStyle, srcB[srcBRow, i]
                )
                if self._gmpool_as_comparable(x) >= self._gmpool_as_comparable(maximum):
                    maximum = x

            # Write the result back, zeroing the rest of the 4x16 block. The
            # accumulating row re-asserts its zero flag only once the last
            # column lands, so the remaining columns of this same pass still
            # read the row as minus infinity rather than as the max of column 0.
            dstVal = self._gmpool_write_dst(maximum, dstStyle)
            if useDst32b:
                dst.setDst32b(dstRow, j, dstVal, validOnLastColumn=True)
                for i in range(1, 4):
                    dst.setDst32b(dstRow + i, j, 0)
            else:
                dst.setDst16b(dstRow, j, dstVal >> 16, validOnLastColumn=True)
                for i in range(1, 4):
                    dst.setDst16b(dstRow + i, j, 0)

        self.optionally_flip_src_banks(issue_thread, flipSrcA, flipSrcB, "GMPOOL")

        # Advance the RWCs
        rwc.applyAddrMod(issue_thread, addrMod)

    def handle_dotpv(self, instruction_info, issue_thread, instr_args):
        dstRow = self._read_dst_field(instruction_info, instr_args)
        addrMod = self._read_addr_mode(instruction_info, instr_args)
        flipSrcA = instr_args["clear_dvalid"] & 0x1
        flipSrcB = instr_args["clear_dvalid"] & 0x2

        self.perform_mvmul(
            issue_thread, dstRow, addrMod, False, flipSrcA, flipSrcB, 4, "DOTPV"
        )

    def handle_mvmul(self, instruction_info, issue_thread, instr_args):
        dstRow = self._read_dst_field(instruction_info, instr_args)
        addrMod = self._read_addr_mode(instruction_info, instr_args)
        broadcastSrcBRow = instr_args["instr_mod19"]
        flipSrcA = instr_args["clear_dvalid"] & 0x1
        flipSrcB = instr_args["clear_dvalid"] & 0x2

        self.perform_mvmul(
            issue_thread,
            dstRow,
            addrMod,
            broadcastSrcBRow,
            flipSrcA,
            flipSrcB,
            8,
            "MVMUL",
        )

    def perform_mvmul(
        self,
        issue_thread,
        dstRow,
        addrMod,
        broadcastSrcBRow,
        flipSrcA,
        flipSrcB,
        numRows,
        opcode=None,
    ):
        stateID = self.backend.getThreadConfigValue(
            issue_thread, "CFG_STATE_ID_StateID"
        )
        rwc = self.getRWC(issue_thread)

        srcAStyle, useDst32b = self.get_dataformat_and_useDst(issue_thread, stateID)
        srcARow, srcBRow, baseDstRow = self.get_base_row_ranges(
            issue_thread, stateID, rwc, broadcastSrcBRow
        )
        # The instruction's own Dst offset sits on top of the RWC/config base;
        # dropping it made every MVMUL/GAPOOL write the base row, so e.g.
        # reduce_tile's REDUCE_SCALAR pooled into Dst row 0 instead of the row 4
        # scratch it then transposes back through SrcB.
        dstRow += baseDstRow
        if broadcastSrcBRow:
            numRows -= 1
        dstRow &= 0x400 - numRows

        fidelityPhase = self.determine_fidelity_phase(issue_thread, rwc)

        if srcAStyle in (DataFormat.BF16, DataFormat.TF32):
            self.perform_mvmul_exact(
                srcARow,
                srcBRow,
                dstRow,
                numRows,
                broadcastSrcBRow,
                srcAStyle,
                useDst32b,
                fidelityPhase,
            )
            self.optionally_flip_src_banks(issue_thread, flipSrcA, flipSrcB, opcode)
            rwc.applyAddrMod(issue_thread, addrMod)
            return

        srcAMatrix = [[0 for _ in range(16)] for _ in range(16)]
        srcBMatrix = [[0 for _ in range(16)] for _ in range(numRows)]
        multipliedMatrix = [[0 for _ in range(16)] for _ in range(numRows)]

        for i in range(numRows):
            for j in range(16):
                srcBVal = self.backend.getSrcB(self.srcBBank)[
                    srcBRow + (0 if broadcastSrcBRow else i), j
                ]
                if srcAStyle == DataFormat.INT8:
                    srcBMatrix[i][j] = self.DataFormatConversions.Int8InSrcToInt8(
                        srcBVal & (0x40FFF if fidelityPhase & 2 else 0x7F0FF)
                    )
                else:
                    srcBValFP32 = self.get_elementwise_fp_src_type(srcAStyle, srcBVal)
                    srcBMatrix[i][j] = self.srcBFidelityBits(srcBValFP32, fidelityPhase)

        for i in range(16):
            for j in range(16):
                srcAVal = self.backend.getSrcA(self.srcABank)[srcARow + i, j]
                if srcAStyle == DataFormat.INT8:
                    srcAMatrix[i][j] = self.DataFormatConversions.Int8InSrcToInt8(
                        srcAVal & (0x41FFF if fidelityPhase & 2 else 0x4E0FF)
                    )
                else:
                    srcAValFP32 = self.get_elementwise_fp_src_type(srcAStyle, srcAVal)
                    srcAMatrix[i][j] = self.srcAFidelityBits(srcAValFP32, fidelityPhase)

        for i in range(numRows):
            for j in range(16):
                multipliedMatrix[i][j] = 0
                for k in range(16):
                    multipliedMatrix[i][j] += srcBMatrix[i][k] * srcAMatrix[k][j]

        for i in range(numRows):
            for j in range(16):
                x = multipliedMatrix[i][j]
                if srcAStyle == DataFormat.INT8:
                    x = multipliedMatrix[i][j]
                    x += self.backend.getDst().getDst32b(dstRow + i, j)
                    self.backend.getDst().setDst32b(
                        dstRow + i, j, DataFormatConversions.FP32ToDstFormatFP32(x)
                    )
                else:
                    self.store_elementwise_fp_result(
                        useDst32b, True, srcAStyle, dstRow, i, j, multipliedMatrix[i][j]
                    )
            if broadcastSrcBRow:
                i += 1

        self.optionally_flip_src_banks(issue_thread, flipSrcA, flipSrcB, opcode)

        # Advance the RWCs
        rwc.applyAddrMod(issue_thread, addrMod)

    @staticmethod
    def _fpu_group_sums(srcAVals, srcBVals, fidelityPhase, expProdAdj):
        """Sixteen fidelity-masked products, reduced to one sum per group of eight.

        This is the hardware's fixed-point datapath, not a real-number one:
        each operand contributes only the mantissa slice its fidelity phase
        selects (five bits of SrcA, seven of SrcB, with ``expProdAdj`` carrying
        the slice's scale), so a product is at most twelve bits. The eight
        products of a group are then aligned to the group's largest exponent —
        anything smaller is rounded off there and then — and summed as
        integers.

        ``srcAVals`` / ``srcBVals`` are FP32 bit patterns (which is what the
        8-bit-exponent Src formats widen to exactly). Returns two
        ``(sign, exponent, mantissa)`` triples, mantissas in the 25-bit form
        :meth:`_fpu_accumulate` expects.
        """
        zeroTermExp = max(expProdAdj, 0)
        signs = [0] * 16
        exps = [zeroTermExp] * 16
        mans = [0] * 16
        for k in range(16):
            a, b = srcAVals[k], srcBVals[k]
            if not (a & 0x7F800000) or not (b & 0x7F800000):
                # A zero exponent is no magnitude at all: the product is zero
                continue
            manA = ((a >> 13) & 0x3FF) | 0x400
            manB = ((b >> 13) & 0x3FF) | 0x400
            manA = ((manA >> 1) & 0x1F) if fidelityPhase & 1 else manA >> 6
            manB = ((manB & 0xF) << 3) if fidelityPhase & 2 else manB >> 4
            signs[k] = (a >> 31) ^ (b >> 31)
            exps[k] = ((a >> 23) & 0xFF) + ((b >> 23) & 0xFF) + expProdAdj
            mans[k] = manA * manB

        sums = []
        for group in range(2):
            lanes = range(8 * group, 8 * group + 8)
            expSum = max(exps[k] for k in lanes)
            if expSum <= 0:
                sums.append((0, 0, 0))
                continue
            manSum = 0
            for k in lanes:
                shift = min(expSum - exps[k], 30)
                man = ((mans[k] << 1) + (1 << shift)) >> (shift + 1)
                manSum += -man if signs[k] else man
            sign = 0
            if manSum < 0:
                sign, manSum = 1, -manSum
            sums.append((sign, expSum, manSum << 13))
        return sums

    @staticmethod
    def _fpu_accumulate(groupSums, dstVal, useDst32b, negOneRenormBug=False):
        """Add the two group sums into Dst, returning the new Dst value as FP32.

        The three terms are aligned to the largest exponent of the three and —
        when Dst is 16-bit — each is rounded to Dst's precision *before* the
        add, which is why a long accumulation drifts differently from one done
        in real numbers. The result is renormalised and rounded once more on
        the way back out.

        ``negOneRenormBug`` models Wormhole's renormalisation of a result whose
        sign/magnitude form is exactly -1: the hardware counts leading *sign*
        bits, and the all-ones two's-complement pattern makes the count come
        out 27 too large. Blackhole fixed it.
        """
        signDst = dstVal >> 31
        expDst = (dstVal >> 23) & 0xFF
        manDst = ((dstVal & 0x7FFFFF) | 0x800000) if expDst else 0

        exp = max(groupSums[0][1], groupSums[1][1], expDst)
        if exp <= 0:
            # Underflows before it is even normalised
            return 0

        terms = []
        for sign, termExp, man in groupSums:
            if termExp < exp:
                shift = exp - termExp
                man = ((man + (1 << (shift - 1)) - sign) >> shift) if shift < 31 else 0
            if not useDst32b:
                man = (man + (1 << 12) - sign) & ~0x1FFF
            terms.append((sign, man))
        if expDst < exp:
            shift = exp - expDst
            manDst = ((manDst + (1 << (shift - 1))) >> shift) if shift < 31 else 0
        if not useDst32b:
            manDst = (manDst + (1 << 12)) & ~0x1FFF

        sign = signDst
        man = manDst
        for termSign, termMan in terms:
            man += termMan if termSign == sign else -termMan
        if man < 0:
            sign, man = sign ^ 1, -man
        if not man:
            # Exact cancellation
            return 0

        shift = (32 - man.bit_length()) - 8  # left-shift that normalises to 24 bits
        if negOneRenormBug and sign and man == 1:
            shift -= 27  # the correct shift is 23; the hardware acts as if it were -4
        exp -= shift
        if not useDst32b:
            shift -= 16
        if shift < 0:
            shift = -shift
            man = (man + (1 << (shift - 1))) >> shift
        else:
            man <<= shift
        if not useDst32b:
            man <<= 16
        if man & 0x1000000:
            exp += 1
        man &= 0x7FFFFF
        if exp <= 0:
            return 0
        if exp >= 255:
            # Dst holds no infinities, so this saturates
            return (sign << 31) | (255 << 23)
        return (sign << 31) | (exp << 23) | man

    # -- Batched form of the two methods above --------------------------------
    #
    # The scalar pair above is the readable port of ttsim's C and stays the
    # reference: ``fpu_accumulate_test.py`` pins it against vectors from ttsim's
    # model *and* fuzzes the two ``_batch`` methods below against it, so the two
    # cannot drift. What runs in anger is the batched pair -- the same
    # arithmetic with the per-lane loops replaced by numpy over a whole MVMUL at
    # once (up to 16 SrcB rows x 16 Dst columns x 16 lanes). Nothing in an MVMUL
    # sequences: every (row, column) accumulation reads and writes its own Dst
    # element, so the whole instruction is one batch.
    #
    # Every intermediate is int64 and each method's docstring carries the bound
    # that keeps it there, because numpy wraps silently where the scalar code's
    # Python ints would simply widen.

    @staticmethod
    def _fpu_group_sums_batch(srcAVals, srcBVals, fidelityPhase, expProdAdj):
        """Batched :meth:`_fpu_group_sums`, one call per MVMUL.

        ``srcAVals`` and ``srcBVals`` are int64 arrays of FP32 bit patterns
        whose **leading** axis is the 16 lanes and whose remaining axes
        broadcast against each other: :meth:`perform_mvmul_exact` passes a
        ``(16, 1, columns)`` block of SrcA against a ``(16, rows, 1)`` block of
        SrcB. Lanes lead because that is the axis being reduced, and numpy
        reduces a leading axis (contiguous blocks, one pass per index) around
        5x faster than a trailing one. Returns ``(signs, exps, mans)``, the same
        triples the scalar version returns, with the lane axis replaced by a
        leading length-2 group axis.

        Widths: the fidelity slices cap manA at 31 and manB at 127, so a product
        is under 2**12; the alignment shift is clamped to 30, so
        ``(man << 1) + (1 << shift)`` stays under 2**31 and the shifted-down
        term is never larger than the product it came from; eight of those sum
        to under 2**15, and 2**15 << 13 is under 2**28.
        """
        nonZero = ((srcAVals & 0x7F800000) != 0) & ((srcBVals & 0x7F800000) != 0)

        manA = ((srcAVals >> 13) & 0x3FF) | 0x400
        manB = ((srcBVals >> 13) & 0x3FF) | 0x400
        manA = ((manA >> 1) & 0x1F) if fidelityPhase & 1 else manA >> 6
        manB = ((manB & 0xF) << 3) if fidelityPhase & 2 else manB >> 4

        signs = np.where(nonZero, (srcAVals >> 31) ^ (srcBVals >> 31), 0)
        exps = np.where(
            nonZero,
            ((srcAVals >> 23) & 0xFF) + ((srcBVals >> 23) & 0xFF) + expProdAdj,
            max(expProdAdj, 0),
        )
        mans = np.where(nonZero, manA * manB, 0)

        # Split the lane axis into two groups of eight, then reduce over the
        # eight (lane k belongs to group k // 8, exactly as the reshape lays it
        # out) to align each group to its own largest exponent and sum it.
        groups = (2, 8) + signs.shape[1:]
        signs = signs.reshape(groups)
        exps = exps.reshape(groups)
        mans = mans.reshape(groups)

        expSum = exps.max(axis=1)
        shift = np.minimum(expSum[:, np.newaxis] - exps, 30)
        aligned = ((mans << 1) + (np.int64(1) << shift)) >> (shift + 1)
        manSum = np.where(signs != 0, -aligned, aligned).sum(axis=1)

        live = expSum > 0
        return (
            np.where(live & (manSum < 0), 1, 0),
            np.where(live, expSum, 0),
            np.where(live, np.abs(manSum) << 13, 0),
        )

    @staticmethod
    def _fpu_accumulate_batch(groupSums, dstVals, useDst32b, negOneRenormBug=False):
        """Batched :meth:`_fpu_accumulate`, one call per MVMUL.

        ``groupSums`` is the ``(signs, exps, mans)`` triple of arrays
        :meth:`_fpu_group_sums_batch` returns, whose leading axis is the group
        of two; ``dstVals`` broadcasts against their remaining axes (a broadcast
        SrcB row gives one set of group sums for every Dst row). Returns the new
        Dst values as FP32 bit patterns. The two groups are combined by index
        rather than by a reduction -- with only two of them a ``np.maximum``
        pair beats calling into numpy's reduction machinery, and
        ``np.minimum(np.maximum(...))`` likewise beats ``np.clip``.

        Widths: an aligned group sum is under 2**28 and an aligned Dst mantissa
        at most 2**24, so the three-term add stays under 2**30 -- which is also
        why ``np.frexp`` on the float64 view of it is an exact ``bit_length``.
        Normalising then brings it back under 2**25.
        """
        gSign, gExp, gMan = groupSums
        signDst = dstVals >> 31
        expDst = (dstVals >> 23) & 0xFF
        manDst = np.where(expDst != 0, (dstVals & 0x7FFFFF) | 0x800000, 0)

        exp = np.maximum(np.maximum(gExp[0], gExp[1]), expDst)

        # Align the two group sums and Dst to the largest of the three
        # exponents. A shift of 31 or more rounds the term away entirely; a
        # 16-bit Dst additionally rounds every term to its precision here,
        # before the add, which is what makes a long accumulation drift.
        shift = exp - gExp
        safeShift = np.minimum(np.maximum(shift, 1), 30)
        rounded = (gMan + (np.int64(1) << (safeShift - 1)) - gSign) >> safeShift
        man = np.where(shift == 0, gMan, np.where(shift < 31, rounded, 0))
        if not useDst32b:
            man = (man + (1 << 12) - gSign) & ~0x1FFF

        shiftDst = exp - expDst
        safeShiftDst = np.minimum(np.maximum(shiftDst, 1), 30)
        roundedDst = (manDst + (np.int64(1) << (safeShiftDst - 1))) >> safeShiftDst
        manDst = np.where(shiftDst == 0, manDst, np.where(shiftDst < 31, roundedDst, 0))
        if not useDst32b:
            manDst = (manDst + (1 << 12)) & ~0x1FFF

        signed = np.where(gSign == signDst, man, -man)
        acc = manDst + signed[0] + signed[1]
        sign = np.where(acc < 0, signDst ^ 1, signDst)
        man = np.abs(acc)

        # Renormalise to 24 bits. np.frexp's exponent is exactly bit_length for
        # a non-negative integer this small; man == 0 gives 0, as Python does.
        shift = (32 - np.frexp(man.astype(np.float64))[1].astype(np.int64)) - 8
        if negOneRenormBug:
            shift = np.where((sign != 0) & (man == 1), shift - 27, shift)
        expOut = exp - shift
        if not useDst32b:
            shift = shift - 16
        rshift = np.minimum(np.maximum(-shift, 1), 30)
        down = (man + (np.int64(1) << (rshift - 1))) >> rshift
        man = np.where(shift < 0, down, man << np.maximum(shift, 0))
        if not useDst32b:
            man = man << 16
        expOut = np.where((man & 0x1000000) != 0, expOut + 1, expOut)
        man = man & 0x7FFFFF

        result = (sign << 31) | (np.minimum(np.maximum(expOut, 0), 254) << 23) | man
        # Dst holds no infinities, so an overflow saturates; an underflow before
        # or after normalising, and an exact cancellation, all give zero.
        result = np.where(expOut >= 255, (sign << 31) | (255 << 23), result)
        return np.where((exp <= 0) | (acc == 0) | (expOut <= 0), 0, result)

    def perform_mvmul_exact(
        self,
        srcARow,
        srcBRow,
        dstRow,
        numRows,
        broadcastSrcBRow,
        srcAStyle,
        useDst32b,
        fidelityPhase,
    ):
        """MVMUL/GAPOOL/DOTPV on the 8-bit-exponent FPU datapath.

        Models the datapath as the hardware builds it — truncated products,
        per-group exponent alignment, and a round into Dst on every
        instruction — rather than evaluating the dot product in real numbers
        and rounding once. That matters as soon as an accumulation runs for
        many instructions: reduce_tile at HiFi4 issues sixteen GAPOOLs into the
        same Dst row, and doing the sums exactly drifts a couple of ULP away
        from what the hardware lands on.

        Both arches share the datapath (verified against ttsim, whose FPU model
        is arch-independent bar the -1 renormalisation quirk
        :meth:`_fpu_accumulate` takes as ``negOneRenormBug``). The FP16 and
        INT8 datapaths keep the real-number model in :meth:`perform_mvmul`;
        only the BF16/TF32 (8-bit exponent) formats, which is what every
        current kernel uses, are modelled exactly.
        """
        expProdAdj = -127
        if fidelityPhase & 1:
            expProdAdj -= 5
        if fidelityPhase & 2:
            expProdAdj -= 7

        toFP32 = (
            DataFormatConversions.TF32InSrcToFP32
            if srcAStyle == DataFormat.TF32
            else DataFormatConversions.BF16InSrcToFP32
        )
        srcA = self.backend.getSrcA(self.srcABank)
        srcB = self.backend.getSrcB(self.srcBBank)
        dst = self.backend.getDst()
        negOneRenormBug = not self.backend.blackhole

        if numRows <= 0:
            return

        # One batch for the whole instruction: lanes on the leading axis (see
        # _fpu_group_sums_batch), SrcA columns on one of the trailing two and
        # SrcB rows on the other, so the products broadcast into
        # (lane, row, column). A broadcast SrcB row means one row's worth of
        # group sums serves every Dst row, so only the distinct rows are
        # gathered and numpy broadcasts the rest.
        #
        # The operands are gathered a rectangle at a time and converted out of
        # the Src storage layout a rectangle at a time too: the ``*InSrcTo*``
        # conversions are pure bit arithmetic, so the same functions the scalar
        # paths call convert a whole block when handed one (see
        # DataFormatConversions). SrcB is transposed because its rows index the
        # 16 lanes, and the copy is deliberate -- every later op wants it
        # contiguous.
        srcAMat = toFP32(srcA.readRows(srcARow, 16))
        srcBLanes = srcB.readRows(srcBRow, 1 if broadcastSrcBRow else numRows)
        srcBMat = toFP32(np.ascontiguousarray(srcBLanes.T))

        # Every Dst element is read before any is written, where the scalar loop
        # interleaved the two. That is safe because each (row, column) touches
        # its own element *and* because a row whose zero flag is clear always
        # holds zeroed data: ZEROACC zeroes on clear, and GMPOOL -- the one
        # writer that holds the flag off mid-row -- always reaches its last
        # column and re-asserts it before the instruction ends.
        dstRows = np.arange(dstRow, dstRow + numRows)
        if useDst32b:
            dstVals = DataFormatConversions.FP32InDstToFP32(dst.getDst32bRows(dstRows))
        else:
            dstVals = (
                DataFormatConversions.BF16InDstToBF16(dst.getDst16bRows(dstRows)) << 16
            )

        # The optional Numba path (tt_sim/pe/tensix/backends/fpu_jit.py) fuses
        # the two passes into one scalar loop nest over the same indices. It is
        # bit-exact with the pair below -- fpu_accumulate_test.py fuzzes them
        # against each other -- and returns None whenever numba is absent,
        # disabled, or the workload is too short to repay compiling it, which
        # is the common case and costs two global reads. int64 is forced at the
        # boundary: the kernel compiles for whatever width it is handed, and
        # every bound in the two docstrings above assumes 64.
        fused = get_fused_mvmul()
        if fused is not None:
            results = fused(
                srcAMat.astype(np.int64, copy=False),
                srcBMat.astype(np.int64, copy=False),
                dstVals.astype(np.int64, copy=False),
                int(fidelityPhase),
                int(expProdAdj),
                bool(useDst32b),
                bool(negOneRenormBug),
            )
        else:
            results = self._fpu_accumulate_batch(
                self._fpu_group_sums_batch(
                    srcAMat[:, np.newaxis, :],
                    srcBMat[:, :, np.newaxis],
                    fidelityPhase,
                    expProdAdj,
                ),
                dstVals,
                useDst32b,
                negOneRenormBug,
            )

        if useDst32b:
            dst.setDst32bRows(
                dstRows, DataFormatConversions.FP32ToDstFormatFP32(results)
            )
        else:
            dst.setDst16bRows(
                dstRows, DataFormatConversions.FP32ToDstFormatBF16(results)
            )

    def get_dataformat_and_useDst(self, issue_thread, stateID):
        # Determine data formats
        if self.getThreadConfigValue(issue_thread, "FP16A_FORCE_Enable"):
            srcAStyle = DataFormat.FP16
            useDst32b = False
            return srcAStyle, useDst32b
        elif self.getConfigValue(stateID, "ALU_ACC_CTRL_INT8_math_enabled"):
            srcAStyle = DataFormat.INT8
            useDst32b = True
            return srcAStyle, useDst32b
        else:
            if self.getConfigValue(stateID, "ALU_FORMAT_SPEC_REG_SrcA_override"):
                srcAFmt = self.getConfigValue(stateID, "ALU_FORMAT_SPEC_REG_SrcA_val")
            else:
                srcAFmt = self.getConfigValue(stateID, "ALU_FORMAT_SPEC_REG0_SrcA")
            srcAFmt = self.implied_srcA_format(issue_thread, srcAFmt)
            if srcAFmt in [
                DataFormat.FP32,
                DataFormat.BF16,
                DataFormat.BFP8,
                DataFormat.BFP4,
                DataFormat.BFP2,
                DataFormat.INT32,
                DataFormat.UINT16,
            ]:
                srcAStyle = DataFormat.BF16
            elif srcAFmt in [
                DataFormat.FP16,
                DataFormat.BFP8_b,
                DataFormat.BFP4_b,
                DataFormat.BFP2_b,
                DataFormat.INT8,
            ]:
                srcAStyle = DataFormat.FP16
            else:
                # SrcAFmt == TF32
                srcAStyle = DataFormat.TF32

            useDst32b = self.getConfigValue(stateID, "ALU_ACC_CTRL_Fp32_enabled")
            return srcAStyle, useDst32b

    def implied_srcA_format(self, issue_thread, configuredFmt):
        """The operand format Blackhole's matrix unit actually uses for SrcA.

        Wormhole always takes it from ALU_FORMAT_SPEC_REG0_SrcA. Blackhole
        instead *implies* it from the format the unpacker last wrote the bank
        in, latched when the bank was handed over; the configured register is
        only consulted when DISABLE_IMPLIED_SRCA_FMT_Base is set.

        This matters for the FP32 path. tt-metal leaves ALU_FORMAT_SPEC_REG0_SrcA
        at FP32 while the unpacker converts to TF32 on the way into Src, and
        SrcA holds TF32 in a 19-bit form. Reading the configured FP32 selects
        the BF16 operand style, which narrows the datum to 7 mantissa bits and
        loses ~2 bits of every copied value; the implied TF32 keeps all 10.
        """
        if not self.backend.blackhole:
            return configuredFmt
        if self.getThreadConfigValue(issue_thread, "DISABLE_IMPLIED_SRCA_FMT_Base"):
            return configuredFmt
        latched = self.backend.getSrcA(self.srcABank).getDataFormat()
        return configuredFmt if latched is None else latched

    def implied_srcB_format(
        self, issue_thread, configuredFmt, disableKey="DISABLE_IMPLIED_SRCB_FMT_Base"
    ):
        """The SrcB-bank counterpart of :meth:`implied_srcA_format`.

        MOVB2D implies *both* of its format selects from the SrcB bank (that is
        not a documentation typo — the instruction's "SrcAFmt" select reads
        ImpliedSrcBFmt), each gated by its own thread-config disable bit, hence
        ``disableKey``. Wormhole has neither register and always uses the
        configured format.
        """
        if not self.backend.blackhole:
            return configuredFmt
        if self.getThreadConfigValue(issue_thread, disableKey):
            return configuredFmt
        latched = self.backend.getSrcB(self.srcBBank).getDataFormat()
        return configuredFmt if latched is None else latched

    def get_base_row_ranges(self, issue_thread, stateID, rwc, broadcastSrcBRow):
        # Determine the row range
        srcARow = rwc.SrcA & 0x38
        srcBRow = rwc.SrcB & (0x3F if broadcastSrcBRow else 0x38)
        dstRow = self.getThreadConfigValue(
            issue_thread, "DEST_TARGET_REG_CFG_MATH_Offset"
        )
        dstRow += rwc.Dst + self.getConfigValue(stateID, "DEST_REGW_BASE_Base")
        return srcARow, srcBRow, dstRow

    def determine_fidelity_phase(self, issue_thread, rwc):
        # Determine the fidelity phase.
        fidelityPhase = rwc.FidelityPhase
        fidelityPhase += self.getThreadConfigValue(issue_thread, "FIDELITY_BASE_Phase")
        fidelityPhase &= 3
        return fidelityPhase

    def optionally_flip_src_banks(self, issue_thread, flipsrca, flipsrcb, opcode=None):
        """Release the Src banks this instruction's ``clear_dvalid`` field names.

        The single release site for the whole unit. ``CLEARDVALID``, ``SETRWC``
        (via ``clear_ab_vld``), and every math op carrying ``clear_dvalid``
        used to each carry their own copy of this five-line sequence, so a rule
        added to one of them silently missed the other three.

        The bank pointer advances whether or not the hand-back happens:
        ``CLR_DVALID_Src{A,B}_Disable`` suppresses the release but not the flip,
        exactly as ttsim's ``math_clear_src_valid`` does.
        """
        if flipsrca:
            if not self.getThreadConfigValue(issue_thread, "CLR_DVALID_SrcA_Disable"):
                self._release_src_bank("SrcA", self.srcABank, issue_thread, opcode)
            self.srcABank ^= 1

        if flipsrcb:
            if not self.getThreadConfigValue(issue_thread, "CLR_DVALID_SrcB_Disable"):
                self._release_src_bank("SrcB", self.srcBBank, issue_thread, opcode)
            self.srcBBank ^= 1

    def _release_src_bank(self, which, bank, issue_thread, opcode):
        """Hand one Src bank back to the unpackers, refusing an unowned release.

        See :class:`SrcDvalidError` for why an unowned release is an error
        rather than a no-op, and ``TT_SIM_DISABLE_DVALID_CHECKS`` for the
        escape hatch.
        """
        src = (self.backend.getSrcA if which == "SrcA" else self.backend.getSrcB)(bank)
        if (
            not _dvalid_checks_disabled
            and src.allowedClient != SrcRegister.SrcClient.MatrixUnit
        ):
            raise SrcDvalidError(
                f"{opcode or 'A Matrix Unit instruction'} from thread {issue_thread} "
                f"cleared dvalid on {which} bank {bank}, but that bank's AllowedClient "
                f"is Unpackers -- the Matrix Unit does not own it, so there is nothing "
                f"to release. Something acquired it and released it twice, or released "
                f"it without acquiring. Set "
                f"{DISABLE_DVALID_CHECK_ENV_VAR}=1 to disable this check."
            )
        src.allowedClient = SrcRegister.SrcClient.Unpackers

    def handle_elementwise_op(
        self,
        stateID,
        issue_thread,
        rwc,
        broadcastSrcBCol0,
        broadcastSrcBRow,
        dstRow,
        addDst,
        flipsrca,
        flipsrcb,
        addrMode,
        op_handler,
        int8_handler,
        fp_handler,
        opcode=None,
    ):
        srcAStyle, useDst32b = self.get_dataformat_and_useDst(issue_thread, stateID)

        srcARow, srcBRow, baseDstRow = self.get_base_row_ranges(
            issue_thread, stateID, rwc, broadcastSrcBRow
        )
        # As in perform_mvmul, the instruction's own Dst offset adds to the base.
        dstRow = (dstRow + baseDstRow) & 0x3F8

        fidelityPhase = self.determine_fidelity_phase(issue_thread, rwc)

        if self.getDiagnosticSettings().reportFPUCalculations():
            print(
                f"FPU: perform element wise op, dst starts at {dstRow}, "
                f"srcA starts at {srcARow} and srcB at {srcBRow} by thread {issue_thread}"
            )

        # Perform the element-wise computation
        for i in range(8):
            for j in range(16):
                srcAVal = self.backend.getSrcA(self.srcABank)[srcARow + i, j]
                srcBVal = self.backend.getSrcB(self.srcBBank)[
                    srcBRow + (0 if broadcastSrcBRow else i),
                    0 if broadcastSrcBCol0 else j,
                ]

                if srcAStyle == DataFormat.INT8:
                    int8_handler(
                        fidelityPhase,
                        addDst,
                        srcAVal,
                        srcBVal,
                        dstRow,
                        i,
                        j,
                        op_handler,
                    )
                else:
                    fp_handler(
                        fidelityPhase,
                        useDst32b,
                        addDst,
                        srcAStyle,
                        srcAVal,
                        srcBVal,
                        dstRow,
                        i,
                        j,
                        op_handler,
                    )

        self.optionally_flip_src_banks(issue_thread, flipsrca, flipsrcb, opcode)

        # Advance the RWCs
        rwc.applyAddrMod(issue_thread, addrMode)

    def elementwise_addsub_int8(
        self, fidelityPhase, addDst, srcA, srcB, dstRow, i, j, op_handler
    ):
        srcAInt = DataFormatConversions.Int8InSrcToInt8(srcA)
        srcBInt = DataFormatConversions.Int8InSrcToInt8(srcB)
        result = op_handler(srcAInt, srcBInt, None)
        if addDst:
            result += self.backend.getDst().getDst32b(dstRow + i, j)

        self.backend.getDst().setDst32b(
            dstRow + i, j, DataFormatConversions.FP32ToDstFormatFP32(result)
        )

    def elementwise_fp_other(
        self,
        fidelityPhase,
        useDst32b,
        addDst,
        srcAStyle,
        srcA,
        srcB,
        dstRow,
        i,
        j,
        op_handler,
    ):
        srcAValFP32 = self.get_elementwise_fp_src_type(srcAStyle, srcA)
        srcBValFP32 = self.get_elementwise_fp_src_type(srcAStyle, srcB)

        result = op_handler(srcAValFP32, srcBValFP32, fidelityPhase)

        self.store_elementwise_fp_result(
            useDst32b, addDst, srcAStyle, dstRow, i, j, result
        )

    def elementwise_mul_int8(
        self,
        fidelityPhase,
        addDst,
        srcA,
        srcB,
        dstRow,
        i,
        j,
        _,
    ):
        srcAValInt = self.DataFormatConversions.Int8InSrcToInt8(
            srcA & (0x41FFF if fidelityPhase & 1 else 0x4E0FF)
        )
        srcBValInt = self.DataFormatConversions.Int8InSrcToInt8(
            srcB & (0x40FFF if fidelityPhase & 2 else 0x7F0FF)
        )

        result = srcAValInt * srcBValInt
        result += self.backend.getDst().getDst32b(dstRow + i, j)

        self.backend.getDst().setDst32b(
            dstRow + i, j, DataFormatConversions.FP32ToDstFormatFP32(result)
        )

    def store_elementwise_fp_result(
        self, useDst32b, addDst, srcAStyle, dstRow, i, j, result
    ):
        if useDst32b:
            # Dst is FP32, regardless of SrcAStyle
            if addDst:
                fp32_bits = DataFormatConversions.FP32InDstToFP32(
                    self.backend.getDst().getDst32b(dstRow + i, j)
                )
                result += conv_to_float(fp32_bits)
            self.backend.getDst().setDst32b(
                dstRow + i,
                j,
                DataFormatConversions.FP32ToDstFormatFP32(conv_to_uint32(result)),
            )
        elif srcAStyle == DataFormat.FP16:
            # Dst is FP16, just like SrcAStyle
            if addDst:
                val = self.backend.getDst().getDst16b(dstRow + i, j)
                fp16_bits = DataFormatConversions.FP16InDstToFP16(val)
                result += conv_to_float(DataFormatConversions.FP16ToFP32(fp16_bits))
            self.backend.getDst().setDst16b(
                dstRow + i,
                j,
                DataFormatConversions.FP32ToDstFormatFP16(conv_to_uint32(result)),
            )
        else:
            # Dst is BF16 (SrcAStyle is either BF16 or TF32)
            if addDst:
                bf16_bits = DataFormatConversions.BF16InDstToBF16(
                    self.backend.getDst().getDst16b(dstRow + i, j)
                )
                result += conv_to_float(bf16_bits << 16)
            self.backend.getDst().setDst16b(
                dstRow + i,
                j,
                DataFormatConversions.FP32ToDstFormatBF16(conv_to_uint32(result)),
            )

    def get_elementwise_fp_src_type(self, srcStyle, src):
        match srcStyle:
            case DataFormat.BF16:
                return conv_to_float(DataFormatConversions.BF16InSrcToFP32(src))
            case DataFormat.FP16:
                return conv_to_float(DataFormatConversions.FP16InSrcToFP32(src))
            case DataFormat.TF32:
                return conv_to_float(DataFormatConversions.TF32InSrcToFP32(src))
            case _:
                raise NotImplementedError()

    def handle_elwmul(self, instruction_info, issue_thread, instr_args):
        stateID = self.backend.getThreadConfigValue(
            issue_thread, "CFG_STATE_ID_StateID"
        )

        rwc = self.getRWC(issue_thread)
        broadcastSrcBCol0 = get_nth_bit(instr_args["instr_mod19"], 0)
        broadcastSrcBRow = get_nth_bit(instr_args["instr_mod19"], 1)
        dstRow = self._read_dst_field(instruction_info, instr_args)
        addDst = True  # Always add for ELWMUL
        addrMode = self._read_addr_mode(instruction_info, instr_args)

        flipsrca = get_nth_bit(instr_args["clear_dvalid"], 0)
        flipsrcb = get_nth_bit(instr_args["clear_dvalid"], 1)

        def mul_handler(srcAValFP32, srcBValFP32, fidelityPhase):
            return self.srcAFidelityBits(
                srcAValFP32, fidelityPhase
            ) * self.srcBFidelityBits(srcBValFP32, fidelityPhase)

        self.handle_elementwise_op(
            stateID,
            issue_thread,
            rwc,
            broadcastSrcBCol0,
            broadcastSrcBRow,
            dstRow,
            addDst,
            flipsrca,
            flipsrcb,
            addrMode,
            mul_handler,
            self.elementwise_mul_int8,
            self.elementwise_fp_other,
            "ELWMUL",
        )

    def handle_elwadd(self, instruction_info, issue_thread, instr_args):
        stateID = self.backend.getThreadConfigValue(
            issue_thread, "CFG_STATE_ID_StateID"
        )

        rwc = self.getRWC(issue_thread)
        broadcastSrcBCol0 = get_nth_bit(instr_args["instr_mod19"], 0)
        broadcastSrcBRow = get_nth_bit(instr_args["instr_mod19"], 1)
        dstRow = self._read_dst_field(instruction_info, instr_args)
        addDst = instr_args["dest_accum_en"]
        addrMode = self._read_addr_mode(instruction_info, instr_args)

        flipsrca = get_nth_bit(instr_args["clear_dvalid"], 0)
        flipsrcb = get_nth_bit(instr_args["clear_dvalid"], 1)

        def add_handler(srcAVal, srcBVal, fidelityPhase=None):
            result = srcAVal + srcBVal

            if fidelityPhase is not None:
                # These divisions are rarely desirable, so software
                # is encouraged to ensure that FidelityPhase == 0
                if fidelityPhase & 1:
                    result /= 32.0
                elif fidelityPhase & 2:
                    result /= 128.0

            return result

        self.handle_elementwise_op(
            stateID,
            issue_thread,
            rwc,
            broadcastSrcBCol0,
            broadcastSrcBRow,
            dstRow,
            addDst,
            flipsrca,
            flipsrcb,
            addrMode,
            add_handler,
            self.elementwise_addsub_int8,
            self.elementwise_fp_other,
            "ELWADD",
        )

    def handle_elwsub(self, instruction_info, issue_thread, instr_args):
        stateID = self.backend.getThreadConfigValue(
            issue_thread, "CFG_STATE_ID_StateID"
        )

        rwc = self.getRWC(issue_thread)
        broadcastSrcBCol0 = get_nth_bit(instr_args["instr_mod19"], 0)
        broadcastSrcBRow = get_nth_bit(instr_args["instr_mod19"], 1)
        dstRow = self._read_dst_field(instruction_info, instr_args)
        addDst = instr_args["dest_accum_en"]
        addrMode = self._read_addr_mode(instruction_info, instr_args)

        flipsrca = get_nth_bit(instr_args["clear_dvalid"], 0)
        flipsrcb = get_nth_bit(instr_args["clear_dvalid"], 1)

        def sub_handler(srcAVal, srcBVal, fidelityPhase=None):
            result = srcAVal - srcBVal

            if fidelityPhase is not None:
                # These divisions are rarely desirable, so software
                # is encouraged to ensure that FidelityPhase == 0
                if fidelityPhase & 1:
                    result /= 32.0
                elif fidelityPhase & 2:
                    result /= 128.0

            return result

        self.handle_elementwise_op(
            stateID,
            issue_thread,
            rwc,
            broadcastSrcBCol0,
            broadcastSrcBRow,
            dstRow,
            addDst,
            flipsrca,
            flipsrcb,
            addrMode,
            sub_handler,
            self.elementwise_addsub_int8,
            self.elementwise_fp_other,
            "ELWSUB",
        )

    def handle_setrwc(self, instruction_info, issue_thread, instr_args):
        rwc = self.getRWC(issue_thread)

        srca = get_nth_bit(instr_args["BitMask"], 0)
        srcb = get_nth_bit(instr_args["BitMask"], 1)
        dst = get_nth_bit(instr_args["BitMask"], 2)
        fidelity = get_nth_bit(instr_args["BitMask"], 3)

        srca_cr = get_nth_bit(instr_args["rwc_cr"], 0)
        srcb_cr = get_nth_bit(instr_args["rwc_cr"], 1)
        dst_cr = get_nth_bit(instr_args["rwc_cr"], 2)
        dst_c_to_cr = get_nth_bit(instr_args["rwc_cr"], 3)

        flipsrca = get_nth_bit(instr_args["clear_ab_vld"], 0)
        flipsrcb = get_nth_bit(instr_args["clear_ab_vld"], 1)

        SrcAVal = instr_args["rwc_a"]
        SrcBVal = instr_args["rwc_b"]
        DstVal = instr_args["rwc_d"]

        if srca:
            if srca_cr:
                SrcAVal += rwc.SrcA_Cr
            rwc.SrcA = SrcAVal
            rwc.SrcA_Cr = SrcAVal

        if srcb:
            if srcb_cr:
                SrcBVal += rwc.SrcB_Cr
            rwc.SrcB = SrcBVal
            rwc.SrcB_Cr = SrcBVal

        if dst or dst_c_to_cr:
            if dst_c_to_cr:
                DstVal += rwc.Dst
            elif dst_cr:
                DstVal += rwc.Dst_Cr
            rwc.Dst = DstVal
            rwc.Dst_Cr = DstVal

        if fidelity:
            rwc.FidelityPhase = 0

        self.optionally_flip_src_banks(issue_thread, flipsrca, flipsrcb, "SETRWC")

    def _read_zeroacc_fields(self, instruction_info, instr_args):
        """Decode ZEROACC's ``dst`` / ``clear_mode`` / ``addr_mode``.

        Blackhole re-lays out three of ZEROACC's fields (per ttsim's
        ``data/{bh,wh}/tensix_isa.json``)::

            field              Wormhole                    Blackhole
            dst / where        9:0                         9:0
            addr_mode          16:15                       16:14
            use_32_bit_mode    packed as clear_mode bit 2  18:18 (own field)
            clear_mode         21:19                       20:19

        The shared instruction table encodes the Wormhole layout, so on
        Blackhole ``addr_mode`` silently drops bit 14 (the ADDR_MOD_3 the
        ``clear_zero_flags`` ZEROACC of `copy_tile` asks for decodes as
        ADDR_MOD_1) and ``use_32_bit_mode`` is read from bit 21, which Blackhole
        leaves unassigned. Blackhole's extra ``addr_mode`` bit also leaks into
        the table's ``dst`` — it runs 14:0 there, the next field starting at 15
        — which is harmless for the modes today's kernels use but not for the
        one-row mode.

        So re-read all three from the raw word on Blackhole and fold
        ``use_32_bit_mode`` back in as ``clear_mode`` bit 2, exactly as ttsim's
        ``TENSIX_EXECUTE_ZEROACC`` does under ``TT_ARCH_VERSION == 1``, after
        which the Wormhole-shaped logic below works unchanged.
        """
        if not self.backend.blackhole:
            return (
                instr_args["dst"],
                instr_args["clear_mode"],
                extract_bits(instr_args["AddrMode"], 2, 0),
            )
        raw = instruction_info["raw_instruction"]
        clear_mode = extract_bits(raw, 2, 19) | (get_nth_bit(raw, 18) << 2)
        return extract_bits(raw, 10, 0), clear_mode, extract_bits(raw, 3, 14)

    def handle_zeroacc(self, instruction_info, issue_thread, instr_args):
        ZEROACC_MODE_ONE_ROW = 0
        ZEROACC_MODE_16_ROWS = 1
        ZEROACC_MODE_HALF_OF_DST = 2
        ZEROACC_MODE_ALL_OF_DST = 3

        imm10, clear_mode, addr_mod = self._read_zeroacc_fields(
            instruction_info, instr_args
        )
        mode = extract_bits(clear_mode, 2, 0)
        useDst32b = extract_bits(clear_mode, 1, 2)

        # Blackhole ZEROACC carries a `clear_zero_flags` bit (raw bit 17, absent
        # from the shared Wormhole-layout argument table). When set, the
        # instruction only re-asserts DEST zero-flags and must NOT clear the
        # data: it is the workaround `copy_tile` issues immediately after
        # unpack-to-dest (a Blackhole HW bug drops the zero-flag clear when
        # unpack-to-dest and a packer ZEROACC land in the same cycle), so the
        # rows it names hold the freshly-unpacked operands and clearing either
        # the flags or the data would wipe them (the SFPU then reads 0). ttsim
        # sets the flags *true* here; the unpack-to-dest that just wrote those
        # rows already did that (every Dst write re-asserts the row's flag), so
        # skipping the whole clear is equivalent. The RWC still advances. See
        # ttsim TT_ARCH_VERSION==1 ZEROACC (clear_zero_flags -> valid=true).
        if self.backend.blackhole and get_nth_bit(
            instruction_info["raw_instruction"], 17
        ):
            if mode == ZEROACC_MODE_ONE_ROW or mode == ZEROACC_MODE_16_ROWS:
                self.getRWC(issue_thread).applyAddrMod(issue_thread, addr_mod)
            return

        if mode == ZEROACC_MODE_ONE_ROW:
            state_id = self.getThreadConfigValue(issue_thread, "CFG_STATE_ID_StateID")
            row = imm10
            row += self.getThreadConfigValue(
                issue_thread, "DEST_TARGET_REG_CFG_MATH_Offset"
            )
            row += self.getRWC(issue_thread).Dst + self.getConfigValue(
                state_id, "DEST_REGW_BASE_Base"
            )

            if (
                self.getConfigValue(state_id, "ALU_ACC_CTRL_Fp32_enabled") == 1
                or self.getConfigValue(state_id, "ALU_ACC_CTRL_INT8_math_enabled") == 1
            ):
                self.getDst().setUndefinedRow(row, True)
            else:
                self.getDst().setUndefinedRow(row)
        else:
            if mode == ZEROACC_MODE_16_ROWS:
                imm10 &= 0xFF
                if self.backend.blackhole and not useDst32b:
                    # On Blackhole the 16-row clear is relative to the math
                    # bank unless DEST_ACCESS_CFG_zeroacc_absolute_tile_mode is
                    # set: a math offset into the high half of DEST moves the
                    # cleared block up by 32 (i.e. 512 rows). ttsim
                    # TENSIX_EXECUTE_ZEROACC case 1. (Its 32-bit sibling, case
                    # 5, shifts by 16 on `dst_offset & 768`; not modelled here
                    # because the useDst32b row mapping below already differs
                    # from ttsim's swizzle, so half of that fix would mislead.)
                    dst_offset = (
                        self.getThreadConfigValue(
                            issue_thread, "DEST_TARGET_REG_CFG_MATH_Offset"
                        )
                        + self.getRWC(issue_thread).Dst
                    )
                    absolute_tile_mode = self.getConfigValue(
                        self.getThreadConfigValue(issue_thread, "CFG_STATE_ID_StateID"),
                        "DEST_ACCESS_CFG_zeroacc_absolute_tile_mode",
                    )
                    if not absolute_tile_mode and (dst_offset & 512):
                        imm10 += 32
                if useDst32b:
                    if imm10 < 32:
                        for i in range(16):
                            self.getDst().setUndefinedRow(imm10 * 16 + i, True)
                else:
                    if imm10 < 64:
                        for i in range(16):
                            self.getDst().setUndefinedRow(imm10 * 16 + i, False)
            elif mode == ZEROACC_MODE_HALF_OF_DST:
                if imm10 & 1:
                    # High half
                    for i in range(512, 1024):
                        self.getDst().setUndefinedRow(i)
                else:
                    # Low half
                    for i in range(512):
                        self.getDst().setUndefinedRow(i)
            elif mode == ZEROACC_MODE_ALL_OF_DST:
                #  Mode == ZEROACC_MODE_ALL_OF_DST
                for i in range(1024):
                    self.getDst().setUndefinedRow(i)
            else:
                raise ValueError(f"Unknown mode {hex(mode)}")

        if mode == ZEROACC_MODE_ONE_ROW or mode == ZEROACC_MODE_16_ROWS:
            self.getRWC(issue_thread).applyAddrMod(issue_thread, addr_mod)

    def srcAFidelityBits(self, x, fidelityPhase):
        x = conv_to_uint32(x)
        if fidelityPhase & 1 == 0:
            # Sign, Exp, implicit 1 of Man, next four Man bits
            return conv_to_float(x & 0xFFF80000)
        else:
            # Isolate the next five Man bits not consumed by prior branch
            return conv_to_float(x - (x & 0xFFF83FFF))

    def srcBFidelityBits(self, x, fidelityPhase):
        x = conv_to_uint32(x)
        if fidelityPhase & 2 == 0:
            # Sign, Exp, implicit 1 of Man, next six Man bits
            return conv_to_float(x & 0xFFFE0000)
        else:
            # Isolate the next four Man bits not consumed by prior branch
            return conv_to_float(x - (x & 0xFFFE1FFF))
