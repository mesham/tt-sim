from tt_sim.pe.tensix.backends.backend_base import TensixBackendUnit
from tt_sim.pe.tensix.registers import SrcRegister
from tt_sim.util.bits import extract_bits, get_nth_bit


class MiscellaneousUnit(TensixBackendUnit):
    """
    The misc unit is mainly concerned with setting the ADC registers to generate
    addresses for the packer and unpacker.

    Based on description and code snippets at
    https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/MiscellaneousUnit.md
    """

    OPCODE_TO_HANDLER = {
        "SETADCXY": "handle_setadcxy",
        "SETADCZW": "handle_setadczw",
        "SETADCXX": "handle_setadcxx",
        "INCADCXY": "handle_incadcxy",
        "INCADCZW": "handle_incadczw",
        "DMANOP": "handle_dmanop",
        "SETADC": "handle_setadc",
        "ADDRCRXY": "handle_addrcrxy",
        "ADDRCRZW": "handle_addrcrzw",
        "SETDVALID": "handle_setdvalid",
    }

    def __init__(self, backend):
        super().__init__(backend, MiscellaneousUnit.OPCODE_TO_HANDLER, "Misc")

    def issueInstruction(self, instruction, from_thread):
        # An occupied IPC group takes nothing; see TensixBackendUnit.is_occupied.
        # This unit is not wired to the cost tables, so it never is.
        if self.busy_until is not None and self.is_occupied(
            self.issue_group(instruction)
        ):
            return self._refuse("unit_busy")
        # Accepts one per thread
        for _, thread_id in self.next_instruction:
            if thread_id == from_thread:
                return self._refuse("issue_slot_taken")
        self.next_instruction.append(
            (
                instruction,
                from_thread,
            )
        )
        return True

    def handle_setdvalid(self, instruction_info, issue_thread, instr_args):
        """Hand the Unpackers' current SrcA/SrcB banks to the Matrix Unit.

        MODELLED ON BOTH ARCHITECTURES, DELIBERATELY, even though the ISA docs
        open Blackhole's functional model with ``UnsupportedFunctionality()``
        and the vendor reference simulator refuses it outright there
        (``ttsim src/tensix.cpp``'s ``TENSIX_EXECUTE_SETDVALID``, which
        implements the whole instruction under ``#if TT_ARCH_VERSION == 0`` and
        raises in the ``#else``). That is the opposite of what
        ``UNMODELLED_BLACKHOLE_INSTRUCTIONS`` does in ``backend.py``, so the
        reasons are recorded rather than left to be re-derived:

        1. REAL BLACKHOLE KERNELS ISSUE IT. ``TTI_SETDVALID(0b10)`` appears in
           ``tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_datacopy.h`` (the
           ROW, SCALAR and COL broadcast paths of the 32-bit unpack-to-dest
           datacopy) and in ``ckernel_debug.h``. The four opcodes tt-sim does
           reject on Blackhole are ones no kernel reaches; this is not one of
           them, and refusing it would break a compute path tt-sim exists to
           run in exchange for nothing.
        2. THE ILL-SPECIFIED PART IS ALREADY MODELLED AS WELL AS IT CAN BE.
           What Blackhole adds to the model is
           ``ImpliedSrc{A,B}Fmt[bank] = UnpredictableValue()`` -- and the same
           paragraph says what the hardware does in practice: "it records a
           stale/held copy of a previous unpack's output format". tt-sim's
           implied format IS that stale copy: ``MatrixUnit.implied_srcA_format``
           / ``implied_srcB_format`` read the format latched on the Src bank by
           the last unpack into it, and nothing here disturbs it. So tt-sim
           already produces the documented in-practice behaviour. An
           unpredictable value is not a value that can be modelled; the honest
           options were this one and refusing, and refusing costs a real kernel
           path.
        3. NO DIFFERENTIAL IS AVAILABLE EITHER WAY. Because the vendor sim
           declines the instruction on Blackhole, ``optests/diff.sh`` cannot
           check tt-sim against it there. Refusing would not buy a check; it
           would only remove a path.

        WHAT IS NOT MODELLED, and is the caveat this docstring exists to carry:

        - The ``UnpredictableValue`` itself. A kernel that depends on the
          implied format after a Blackhole ``SETDVALID`` is relying on
          behaviour the docs tell it not to rely on; tt-sim will give it the
          previous unpack's format, silently and deterministically, and
          hardware need not.
        - The instruction's own precondition. Handing a bank to the Matrix Unit
          that it already owns is ``NonContractualBehavior``
          (``TTSIM_VERIFY(!(p_tensix->src_a_valid & (1 << unpack_bank)), ...)``)
          and tt-sim does it without complaint. That is exactly what
          ``perfbench/tensixbench``'s original per-thread setup did, and it
          produced a 12x apparent matrix-unit slowdown on silicon that took an
          experiment to retract -- see
          docs/plans/matrix-unit-thread-contention.md. tt-sim has no per-bank
          valid bit to check it against today; adding one is the fix, and it
          would want a Wormhole-side audit first because a false positive here
          fires on correct kernels.
        Both ``SRCA_SET_Base`` and ``SRCB_SET_Base`` are read below, as
        ``SETDVALID.md``'s model does. The second is always zero in practice --
        the vendor sim notes it "is not instantiated and errors on write" -- so
        reading it costs nothing and keeps this handler a transcription of the
        published model rather than a transcription plus a shortcut.
        """
        flipSrcA = instr_args["setvalid"] & 0x1
        flipSrcB = (instr_args["setvalid"] >> 1) & 0x1

        if flipSrcA:
            self.backend.getSrcA(
                self.backend.unpacker_units[0].srcBank
            ).allowedClient = SrcRegister.SrcClient.MatrixUnit
            self.backend.unpacker_units[0].srcBank ^= 1
            # Indexed by thread, exactly as the unpacker's own bank-flip does
            # (``UnPackerUnit.handle_give_src_to_fpu``). Assigning the bare
            # attribute replaces the per-thread list with a scalar and the next
            # ``UNPACR`` dies with "'int' object does not support item
            # assignment" -- reachable from any kernel that issues SETDVALID
            # before an unpack.
            self.backend.unpacker_units[0].srcRow[issue_thread] = (
                self.backend.getThreadConfigValue(issue_thread, "SRCA_SET_Base") << 4
            )
        if flipSrcB:
            self.backend.getSrcB(
                self.backend.unpacker_units[1].srcBank
            ).allowedClient = SrcRegister.SrcClient.MatrixUnit
            self.backend.unpacker_units[1].srcBank ^= 1
            self.backend.unpacker_units[1].srcRow[issue_thread] = (
                self.backend.getThreadConfigValue(issue_thread, "SRCB_SET_Base") << 4
            )

    def handle_dmanop(self, instruction_info, issue_thread, instr_args):
        # This is a nop (but in documentation says for the scalar unit, but it is
        # directed to the misc unit for some reason)
        pass

    def handle_addrcrzw(self, instruction_info, issue_thread, instr_args):
        # "CR" is carriage-return: the instruction *adds* its operand to the
        # carriage-return register and then snaps the counter to it (see
        # ADDRCRZW.md's `ADC_.Channel[0].Z_Cr += Z0Inc, ... .Z = ... .Z_Cr`).
        # Assigning instead of adding pinned the carriage return at its first
        # value, so a loop that walks rows by repeating ADDRCRZW never got past
        # row one. Same shape as ADDRCRXY below.
        def apply_to(adc_channel, enables, Z0Inc, W0Inc, Z1Inc, W1Inc):
            if get_nth_bit(enables, 0):
                adc_channel.Channel[0].Z_Cr += Z0Inc
                adc_channel.Channel[0].Z = adc_channel.Channel[0].Z_Cr

            if get_nth_bit(enables, 1):
                adc_channel.Channel[0].W_Cr += W0Inc
                adc_channel.Channel[0].W = adc_channel.Channel[0].W_Cr

            if get_nth_bit(enables, 2):
                adc_channel.Channel[1].Z_Cr += Z1Inc
                adc_channel.Channel[1].Z = adc_channel.Channel[1].Z_Cr

            if get_nth_bit(enables, 3):
                adc_channel.Channel[1].W_Cr += W1Inc
                adc_channel.Channel[1].W = adc_channel.Channel[1].W_Cr

        Z0Inc = instr_args["Ch0_X"]
        W0Inc = instr_args["Ch0_Y"]
        Z1Inc = instr_args["Ch1_X"]
        W1Inc = extract_bits(instr_args["Ch1_Y"], 3, 0)
        enables = instr_args["BitMask"]
        threadOverride = extract_bits(instr_args["Ch1_Y"], 2, 3)

        whichThread = issue_thread if threadOverride == 0 else threadOverride - 1

        if get_nth_bit(instr_args["CntSetMask"], 0):
            apply_to(
                self.backend.getADC(whichThread).Unpacker[0],
                enables,
                Z0Inc,
                W0Inc,
                Z1Inc,
                W1Inc,
            )
        if get_nth_bit(instr_args["CntSetMask"], 1):
            apply_to(
                self.backend.getADC(whichThread).Unpacker[1],
                enables,
                Z0Inc,
                W0Inc,
                Z1Inc,
                W1Inc,
            )
        if get_nth_bit(instr_args["CntSetMask"], 2):
            apply_to(
                self.backend.getADC(whichThread).Packers,
                enables,
                Z0Inc,
                W0Inc,
                Z1Inc,
                W1Inc,
            )

    def handle_addrcrxy(self, instruction_info, issue_thread, instr_args):
        # Accumulates into the carriage-return register, per ADDRCRXY.md:
        # `ADC_.Channel[0].Y_Cr += Y0Inc, ADC_.Channel[0].Y = ADC_.Channel[0].Y_Cr`.
        # `pack_untilize_dest` walks the 8 output rows each packer owns by
        # issuing one ADDRCRXY(Y0Inc=1) per row ("Read new row in the tile" in
        # llk_pack_untilize.h); assigning rather than adding left Y_Cr stuck at
        # 1 from the second row on, so every later row re-read the same Dst row.
        def apply_to(adc_channel, enables, X0Inc, Y0Inc, X1Inc, Y1Inc):
            if get_nth_bit(enables, 0):
                adc_channel.Channel[0].X_Cr += X0Inc
                adc_channel.Channel[0].X = adc_channel.Channel[0].X_Cr

            if get_nth_bit(enables, 1):
                adc_channel.Channel[0].Y_Cr += Y0Inc
                adc_channel.Channel[0].Y = adc_channel.Channel[0].Y_Cr

            if get_nth_bit(enables, 2):
                adc_channel.Channel[1].X_Cr += X1Inc
                adc_channel.Channel[1].X = adc_channel.Channel[1].X_Cr

            if get_nth_bit(enables, 3):
                adc_channel.Channel[1].Y_Cr += Y1Inc
                adc_channel.Channel[1].Y = adc_channel.Channel[1].Y_Cr

        X0Inc = instr_args["Ch0_X"]
        Y0Inc = instr_args["Ch0_Y"]
        X1Inc = instr_args["Ch1_X"]
        Y1Inc = extract_bits(instr_args["Ch1_Y"], 3, 0)
        enables = instr_args["BitMask"]
        threadOverride = extract_bits(instr_args["Ch1_Y"], 2, 3)

        whichThread = issue_thread if threadOverride == 0 else threadOverride - 1

        if get_nth_bit(instr_args["CntSetMask"], 0):
            apply_to(
                self.backend.getADC(whichThread).Unpacker[0],
                enables,
                X0Inc,
                Y0Inc,
                X1Inc,
                Y1Inc,
            )
        if get_nth_bit(instr_args["CntSetMask"], 1):
            apply_to(
                self.backend.getADC(whichThread).Unpacker[1],
                enables,
                X0Inc,
                Y0Inc,
                X1Inc,
                Y1Inc,
            )
        if get_nth_bit(instr_args["CntSetMask"], 2):
            apply_to(
                self.backend.getADC(whichThread).Packers,
                enables,
                X0Inc,
                Y0Inc,
                X1Inc,
                Y1Inc,
            )

    def handle_incadczw(self, instruction_info, issue_thread, instr_args):
        def apply_to(adc_channel, Z0Inc, W0Inc, Z1Inc, W1Inc):
            adc_channel.Channel[0].Z += Z0Inc
            adc_channel.Channel[0].W += W0Inc
            adc_channel.Channel[1].Z += Z1Inc
            adc_channel.Channel[1].W += W1Inc

        Z0Inc = instr_args["Ch0_X"]
        W0Inc = instr_args["Ch0_Y"]
        Z1Inc = instr_args["Ch1_X"]
        W1Inc = extract_bits(instr_args["Ch1_Y"], 3, 0)

        threadOverride = extract_bits(instr_args["Ch1_Y"], 2, 3)
        whichThread = issue_thread if threadOverride == 0 else threadOverride - 1

        if get_nth_bit(instr_args["CntSetMask"], 0):
            apply_to(
                self.backend.getADC(whichThread).Unpacker[0], Z0Inc, W0Inc, Z1Inc, W1Inc
            )
        if get_nth_bit(instr_args["CntSetMask"], 1):
            apply_to(
                self.backend.getADC(whichThread).Unpacker[1], Z0Inc, W0Inc, Z1Inc, W1Inc
            )
        if get_nth_bit(instr_args["CntSetMask"], 2):
            apply_to(
                self.backend.getADC(whichThread).Packers, Z0Inc, W0Inc, Z1Inc, W1Inc
            )

    def handle_incadcxy(self, instruction_info, issue_thread, instr_args):
        def apply_to(adc_channel, X0Inc, Y0Inc, X1Inc, Y1Inc):
            adc_channel.Channel[0].X += X0Inc
            adc_channel.Channel[0].Y += Y0Inc
            adc_channel.Channel[1].X += X1Inc
            adc_channel.Channel[1].Y += Y1Inc

        X0Inc = instr_args["Ch0_X"]
        Y0Inc = instr_args["Ch0_Y"]
        X1Inc = instr_args["Ch1_X"]
        Y1Inc = extract_bits(instr_args["Ch1_Y"], 3, 0)

        threadOverride = extract_bits(instr_args["Ch1_Y"], 2, 3)
        whichThread = issue_thread if threadOverride == 0 else threadOverride - 1

        if get_nth_bit(instr_args["CntSetMask"], 0):
            apply_to(
                self.backend.getADC(whichThread).Unpacker[0], X0Inc, Y0Inc, X1Inc, Y1Inc
            )
        if get_nth_bit(instr_args["CntSetMask"], 1):
            apply_to(
                self.backend.getADC(whichThread).Unpacker[1], X0Inc, Y0Inc, X1Inc, Y1Inc
            )
        if get_nth_bit(instr_args["CntSetMask"], 2):
            apply_to(
                self.backend.getADC(whichThread).Packers, X0Inc, Y0Inc, X1Inc, Y1Inc
            )

    def handle_setadcxx(self, instruction_info, issue_thread, instr_args):
        def apply_to(adc_channel, X0Val, X1Val):
            adc_channel.Channel[0].X = X0Val
            adc_channel.Channel[0].X_Cr = X0Val
            adc_channel.Channel[1].X = X1Val
            adc_channel.Channel[1].X_Cr = X1Val

        X0Val = instr_args["x_start"]
        X1Val = instr_args["x_end2"]

        if get_nth_bit(instr_args["CntSetMask"], 0):
            apply_to(self.backend.getADC(issue_thread).Unpacker[0], X0Val, X1Val)
        if get_nth_bit(instr_args["CntSetMask"], 1):
            apply_to(self.backend.getADC(issue_thread).Unpacker[1], X0Val, X1Val)
        if get_nth_bit(instr_args["CntSetMask"], 2):
            apply_to(self.backend.getADC(issue_thread).Packers, X0Val, X1Val)

    def handle_setadc(self, instruction_info, issue_thread, instr_args):
        def apply_to(adc_channel, xyzw, newValue):
            match xyzw:
                case 0:
                    adc_channel.X = newValue
                    adc_channel.X_Cr = newValue
                case 1:
                    adc_channel.Y = newValue
                    adc_channel.Y_Cr = newValue
                case 2:
                    adc_channel.Z = newValue
                    adc_channel.Z_Cr = newValue
                case 3:
                    adc_channel.W = newValue
                    adc_channel.W_Cr = newValue

        newValue = instr_args["Value"]
        threadOverride = newValue >> 16
        whichThread = issue_thread if threadOverride == 0 else threadOverride - 1

        channelIndex = instr_args["ChannelIndex"]
        xyzw = instr_args["DimensionIndex"]

        if get_nth_bit(instr_args["CntSetMask"], 0):
            apply_to(
                self.backend.getADC(whichThread).Unpacker[0].Channel[channelIndex],
                xyzw,
                newValue,
            )
        if get_nth_bit(instr_args["CntSetMask"], 1):
            apply_to(
                self.backend.getADC(whichThread).Unpacker[1].Channel[channelIndex],
                xyzw,
                newValue,
            )
        if get_nth_bit(instr_args["CntSetMask"], 2):
            apply_to(
                self.backend.getADC(whichThread).Packers.Channel[channelIndex],
                xyzw,
                newValue,
            )

    def handle_setadczw(self, instruction_info, issue_thread, instr_args):
        def apply_to(adc_channel, enables, X0Val, Y0Val, X1Val, Y1Val):
            if get_nth_bit(enables, 0):
                adc_channel.Channel[0].Z = X0Val
                adc_channel.Channel[0].W_Cr = X0Val

            if get_nth_bit(enables, 1):
                adc_channel.Channel[0].Z = Y0Val
                adc_channel.Channel[0].W_Cr = Y0Val

            if get_nth_bit(enables, 2):
                adc_channel.Channel[1].Z = X1Val
                adc_channel.Channel[1].W_Cr = X1Val

            if get_nth_bit(enables, 3):
                adc_channel.Channel[1].Z = Y1Val
                adc_channel.Channel[1].W_Cr = Y1Val

        threadOverride = extract_bits(instr_args["Ch1_Y"], 2, 3)
        whichThread = issue_thread if threadOverride == 0 else threadOverride - 1

        X0Val = instr_args["Ch0_X"]
        Y0Val = instr_args["Ch0_Y"]
        X1Val = instr_args["Ch1_X"]
        Y1Val = extract_bits(instr_args["Ch1_Y"], 3, 0)
        enables = instr_args["BitMask"]

        if get_nth_bit(instr_args["CntSetMask"], 0):
            apply_to(
                self.backend.getADC(whichThread).Unpacker[0],
                enables,
                X0Val,
                Y0Val,
                X1Val,
                Y1Val,
            )
        if get_nth_bit(instr_args["CntSetMask"], 1):
            apply_to(
                self.backend.getADC(whichThread).Unpacker[1],
                enables,
                X0Val,
                Y0Val,
                X1Val,
                Y1Val,
            )
        if get_nth_bit(instr_args["CntSetMask"], 2):
            apply_to(
                self.backend.getADC(whichThread).Packers,
                enables,
                X0Val,
                Y0Val,
                X1Val,
                Y1Val,
            )

    def handle_setadcxy(self, instruction_info, issue_thread, instr_args):
        def apply_to(adc_channel, enables, X0Val, Y0Val, X1Val, Y1Val):
            if get_nth_bit(enables, 0):
                adc_channel.Channel[0].X = X0Val
                adc_channel.Channel[0].X_Cr = X0Val

            if get_nth_bit(enables, 1):
                adc_channel.Channel[0].Y = Y0Val
                adc_channel.Channel[0].Y_Cr = Y0Val

            if get_nth_bit(enables, 2):
                adc_channel.Channel[1].X = X1Val
                adc_channel.Channel[1].X_Cr = X1Val

            if get_nth_bit(enables, 3):
                adc_channel.Channel[1].Y = Y1Val
                adc_channel.Channel[1].Y_Cr = Y1Val

        threadOverride = extract_bits(instr_args["Ch1_Y"], 2, 3)
        whichThread = issue_thread if threadOverride == 0 else threadOverride - 1

        X0Val = instr_args["Ch0_X"]
        Y0Val = instr_args["Ch0_Y"]
        X1Val = instr_args["Ch1_X"]
        Y1Val = extract_bits(instr_args["Ch1_Y"], 3, 0)
        enables = instr_args["BitMask"]

        if get_nth_bit(instr_args["CntSetMask"], 0):
            apply_to(
                self.backend.getADC(whichThread).Unpacker[0],
                enables,
                X0Val,
                Y0Val,
                X1Val,
                Y1Val,
            )
        if get_nth_bit(instr_args["CntSetMask"], 1):
            apply_to(
                self.backend.getADC(whichThread).Unpacker[1],
                enables,
                X0Val,
                Y0Val,
                X1Val,
                Y1Val,
            )
        if get_nth_bit(instr_args["CntSetMask"], 2):
            apply_to(
                self.backend.getADC(whichThread).Packers,
                enables,
                X0Val,
                Y0Val,
                X1Val,
                Y1Val,
            )
