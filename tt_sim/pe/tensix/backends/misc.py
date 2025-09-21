from tt_sim.pe.tensix.backends.backend_base import TensixBackendUnit
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
    }

    def __init__(self, backend):
        super().__init__(backend, MiscellaneousUnit.OPCODE_TO_HANDLER, "Misc")

    def issueInstruction(self, instruction, from_thread):
        # Accepts one per thread
        for _, thread_id in self.next_instruction:
            if thread_id == from_thread:
                return False
        self.next_instruction.append(
            (
                instruction,
                from_thread,
            )
        )
        return True

    def handle_dmanop(self, instruction_info, issue_thread, instr_args):
        # This is a nop (but in documentation says for the scalar unit, but it is
        # directed to the misc unit for some reason)
        pass

    def handle_addrcrzw(self, instruction_info, issue_thread, instr_args):
        def apply_to(adc_channel, enables, Z0Inc, W0Inc, Z1Inc, W1Inc):
            if get_nth_bit(enables, 0):
                adc_channel.Channel[0].Z_Cr = Z0Inc
                adc_channel.Channel[0].Z = Z0Inc

            if get_nth_bit(enables, 1):
                adc_channel.Channel[0].W_Cr = W0Inc
                adc_channel.Channel[0].W = W0Inc

            if get_nth_bit(enables, 2):
                adc_channel.Channel[1].Z_Cr = Z1Inc
                adc_channel.Channel[1].Z = Z1Inc

            if get_nth_bit(enables, 3):
                adc_channel.Channel[1].W_Cr = W1Inc
                adc_channel.Channel[1].W = W1Inc

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
        def apply_to(adc_channel, enables, X0Inc, Y0Inc, X1Inc, Y1Inc):
            if get_nth_bit(enables, 0):
                adc_channel.Channel[0].X_Cr = X0Inc
                adc_channel.Channel[0].X = X0Inc

            if get_nth_bit(enables, 1):
                adc_channel.Channel[0].Y_Cr = Y0Inc
                adc_channel.Channel[0].Y = Y0Inc

            if get_nth_bit(enables, 2):
                adc_channel.Channel[1].X_Cr = X1Inc
                adc_channel.Channel[1].X = X1Inc

            if get_nth_bit(enables, 3):
                adc_channel.Channel[1].Y_Cr = Y1Inc
                adc_channel.Channel[1].Y = Y1Inc

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
