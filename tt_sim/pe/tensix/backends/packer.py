from enum import IntEnum
from math import ceil

from tt_sim.pe.tensix.backends.backend_base import (
    DATA_FORMAT_TO_BITS,
    DATA_FORMAT_TO_NAME,
    DataFormat,
    TensixBackendUnit,
)
from tt_sim.pe.tensix.util import DataFormatConversions
from tt_sim.perf.model import unit_cost_model
from tt_sim.util.bits import extract_bits, get_nth_bit
from tt_sim.util.conversion import conv_to_bytes


class PackerUnit(TensixBackendUnit):
    """
    Packer backend unit, note that currently this is only running one packer to do the
    actual packing (all four calculate input and output addresses) as the logic doesn't
    share the packing but instead duplicates it out.

    Based on description and code snippets at
    https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/UNPACR.md
    """

    OPCODE_TO_HANDLER = {"PACR": "handle_pacr"}

    packer0InitialAddr = 0
    datastreamNeedsNewAddr = True
    byteAddress = 0

    class PackerI:
        def __init__(self):
            self.inputNumDatums = 0
            self.inputSource = 0
            self.inputSourceAddr = 0
            self.inputSourceStride = 0
            self.inDataFormat = None
            # PCK_DEST_RD_CTRL_Read_32b_data: whether Dst is read 32 bits at a
            # time. This is *not* implied by inDataFormat -- see
            # PackerUnit.handle_pacr.
            self.readDst32b = False
            self.byteAddress = 0
            # The first datastream (before any Last/Flush has run) still needs a
            # fresh output address computed from config; without this the very
            # first tile packs from address 0 instead of the CB base.
            self.datastreamNeedsNewAddr = True
            self.outBytes = 0
            self.outDataFormat = None
            # Tile-pack-generator row/column counters, used to pick the edge
            # mask for each datum (see PackerUnit.edge_masks_for_pacr).
            self.tpgX = 0
            self.tpgY = 0

    class InputSource(IntEnum):
        L1 = 1
        DST = 2

    def __init__(self, backend):
        self.packerI = [PackerUnit.PackerI() for i in range(4)]
        super().__init__(backend, PackerUnit.OPCODE_TO_HANDLER, "Packer")
        # Phase 5 of docs/plans/event-driven-pump.md. ``None`` unless
        # TT_SIM_COST_MODEL is set, and even then it charges only what the ISA
        # docs actually publish for this unit: PACR's *issue* cost of one cycle
        # ("at most one of these instructions can be started per cycle"). The
        # time the packers take to drain the work afterwards has no published
        # per-op figure anywhere -- not the ISA docs, not tt-metal, not the LLK
        # -- so it is charged nothing rather than estimated. That is the
        # largest deliberate gap in the cost tables; see the PACK unit's note
        # in tensix_instruction_costs.yaml and "What is missing" in
        # docs/plans/cost-model.md.
        self.cost_model = unit_cost_model(
            "PACK", "blackhole" if backend.blackhole else "wormhole"
        )

    def participating_packers(self, packMask):
        """Which packers this PACR actually drives.

        On **Wormhole** the PACR field in bits 11:8 is `PackSel`, a bit mask
        selecting up to four independent packers (0 meaning "packer 0"), each
        with its own configuration section: packers 0/1 read `THCON_SEC0_REG1`
        / `THCON_SEC0_REG8`, packers 2/3 read `THCON_SEC1_REG1` /
        `THCON_SEC1_REG8` (see `getIPackerConfig`).

        On **Blackhole** there is only **one** packer. The same instruction
        bits are `read_intf_sel`, selecting which of that packer's four Dst
        *read interfaces* are active — not which packers run. Its whole
        configuration comes from `THCON_SEC0_REG1`; the `SEC0_REG8` /
        `SEC1_*` sections are not packer configuration on this architecture and
        tt-metal never writes them. Treating the mask as a packer selection
        therefore both duplicated the pack to a bogus L1 address and consulted
        registers that read as zero — which the `Disable_zero_compress` check
        below, inverted in the naming, read as "zero compression enabled" and
        refused. ttsim is explicit about this: its `TENSIX_EXECUTE_PACR` has
        `constexpr uint32_t n_packers = 1` under `TT_ARCH_VERSION == 1` and
        indexes only `THCON_SEC0_REG1`.
        """
        if self.backend.blackhole:
            return (0,)
        return tuple(i for i in range(4) if get_nth_bit(packMask, i)) or (0,)

    def dst_read_interfaces(self, packMask):
        """Blackhole: the Dst read interfaces `read_intf_sel` selects.

        Interface `k` reads Dst row `base + k`, so the interfaces a PACR
        selects fix both how many datums it moves (16 per interface) and which
        rows they come from: `0b1111`/`0` walks four contiguous rows,
        `0b0101` the even rows of a pair and `0b1010` the odd ones — which is
        how the tilize pack MOP interleaves two Dst halves into one tile.
        """
        return [k for k in range(4) if get_nth_bit(packMask, k)] or [0, 1, 2, 3]

    #: Dst row stride between successive read interfaces in
    #: ``DST_ACCESS_STRIDED_MODE``. See ``_read_dst_access_mode``.
    STRIDED_DST_ROW_STRIDE = 16

    #: Blackhole PACR fields that the shared (Wormhole) argument table has no
    #: entry for and that nothing here models, as
    #: ``(name, bit span, width, start bit)``. See
    #: :meth:`_check_unmodelled_bh_fields`.
    BH_UNMODELLED_FIELDS = (
        ("ctxt_ctrl", "3:2", 2, 2),
        ("addr_cnt_context", "14:13", 2, 13),
        ("row_pad_zero", "20:18", 3, 18),
        ("cfg_context", "22:21", 2, 21),
    )

    def _check_unmodelled_bh_fields(self, instruction_info):
        """Refuse a Blackhole PACR that sets a field tt-sim does not decode.

        Blackhole's PACR encodes four fields on top of Wormhole's, and the
        instruction table in ``tensix_instructions.yaml`` is the *Wormhole*
        one — so none of them has a name here and none is ever read. Nothing
        is mis-decoded as a result (see below), but a kernel that set one
        would have it silently ignored, which is the failure shape everything
        else on this unit refuses outright. ttsim refuses all four the same
        way, as ``MissingSpecification`` under ``TT_ARCH_VERSION == 1`` at the
        top of ``TENSIX_EXECUTE_PACR``; this mirrors that.

        Why nothing is currently mis-decoded, given that the shared table
        hands each argument every bit up to the next one's ``start_bit``
        (``TensixInstructions.getInstructionInfo``):

        * ``ctxt_ctrl`` (3:2) lands inside ``Flush``'s decoded 3:1, and
          ``handle_pacr`` takes only ``get_nth_bit(Flush, 0)`` — raw bit 1.
        * ``addr_cnt_context`` (14:13) lands inside ``ZeroWrite``'s decoded
          14:12, likewise consumed as ``get_nth_bit(ZeroWrite, 0)`` — raw
          bit 12.
        * ``row_pad_zero`` (20:18) and ``cfg_context`` (22:21) land inside
          ``AddrMode``'s decoded 23:15, which on Blackhole is not read from
          the decoded argument at all — :meth:`_read_addr_mode` re-reads raw
          16:15 precisely because that span is shared.

        That last one is why these are read from ``raw_instruction`` here and
        not from ``instr_args``: two separate bugs have already come out of
        the 23:15 span, and a decoded argument is the wrong thing to look at.
        """
        if not self.backend.blackhole:
            # Wormhole's PACR is correct today and these bit positions mean
            # something else there (Flush, ZeroWrite and AddrMode's tail).
            return
        raw = instruction_info["raw_instruction"]
        for name, span, width, start in self.BH_UNMODELLED_FIELDS:
            value = extract_bits(raw, width, start)
            if value:
                raise NotImplementedError(
                    f"Blackhole PACR sets {name}={value} (raw bits {span}), which is "
                    "not modelled; tt-sim decodes PACR through the shared Wormhole "
                    "argument table, which has no entry for this field, so it would "
                    "otherwise be silently ignored (ttsim refuses it too, as "
                    "MissingSpecification in TENSIX_EXECUTE_PACR)"
                )

    def _read_addr_mode(self, instruction_info, instr_args):
        """PACR's 2-bit ``addr_mode`` (raw 16:15).

        ``AddrMode`` is the *last* argument in the shared (Wormhole) table, so
        the decoder gives it every bit up to the opcode — 23:15. That is
        harmless on Wormhole, where nothing above bit 16 is encoded, but
        Blackhole packs three more fields into that span: ``dst_access_mode``
        (17), ``row_pad_zero`` (20:18) and ``cfg_context`` (22:21). A
        ``pack_untilize`` PACR sets ``dst_access_mode``, so its ``ADDR_MOD_1``
        decoded as section 5 — a section no LLK programs — and the pack Y
        counter stopped advancing: every one of the MOP's 16 row-closing PACRs
        re-read the same Dst face row, so one output row was replicated down the
        whole tile.
        """
        if self.backend.blackhole:
            return extract_bits(instruction_info["raw_instruction"], 2, 15)
        return instr_args["AddrMode"]

    def _read_dst_access_mode(self, instruction_info, packMask):
        """Blackhole PACR's ``dst_access_mode`` (raw bit 17).

        The shared instruction table stops at Wormhole's ``AddrMode`` (start bit
        15), so this Blackhole-only field is re-read from the raw word, as every
        other moved/added Blackhole operand is (``MatrixUnit._read_addr_mode``,
        ``VectorUnit._read_sfpu_addr_mode``, ...).

        ``0`` is ``DST_ACCESS_NORMAL_MODE``, where read interface ``k`` supplies
        its 16 datums from Dst row ``base + k`` (see ``dst_read_interfaces``).
        ``1`` is ``DST_ACCESS_STRIDED_MODE``, where the interfaces step by 16
        rows instead of 1 — interface ``k`` reads row ``base + 16*k`` — so a
        PACR gathers the *same* row of successive 16-row faces rather than
        successive rows of one face. That is what makes a row-major
        ``pack_untilize_dest`` a plain pack: its MOP issues 16 two-interface
        PACRs per face pair, and each emits one 32-datum output row spanning
        face ``f`` and face ``f+1``.

        Sources, in order of authority: the ISA docs have no BlackholeA0 PACR
        or Packers chapter at all (BlackholeA0/.../Dst.md links to pages that do
        not exist), so all they settle is that ``DEST_ACCESS_CFG_remap_addrs``
        and ``DEST_ACCESS_CFG_swizzle_32b`` "also affect how packers address
        Dst" beyond ``Adj16``/``Adj32``. The stride of 16 itself comes from
        ttsim (``TENSIX_EXECUTE_PACR``: ``row = pack_row + 16*(i / ROW_SIZE)``,
        "remap and swizzle cause stride to be 16 rows and not 8 here") and is
        corroborated by tt-metal's ``_llk_math_reconfig_remap_``, which sets
        both config bits together and documents them as "needed for enabling
        stride of 16". Note the stride is 16 in *logical* Dst rows: tt-sim, like
        hardware and unlike ttsim, applies ``Adj16``/``Adj32`` to every Dst
        access, so the hardware's physical stride of 8 through the remapped
        layout is the logical stride of 16 modelled here.

        Strided *without* those two config bits is a different matter, and it is
        refused below rather than guessed. Nothing specifies it: there is no
        Blackhole PACR page, and ttsim declines the identical combination
        ("We currently require strided mode to be tied to the swizzle_32b and
        remap_addrs features"). Reaching it means the kernel is out of contract
        rather than that a mode is missing — see the refusal's own note.
        """
        if not self.backend.blackhole:
            return 0
        strided = get_nth_bit(instruction_info["raw_instruction"], 17)
        if not strided:
            return 0
        dst = self.backend.getDst()
        if not (dst.dest_remap_addrs and dst.dest_swizzle_32b):
            # ttsim ties strided mode to both DEST_ACCESS_CFG bits and refuses
            # otherwise; without the remap the stride is not 16 and there is no
            # reference for what it would be.
            #
            # The message names the cause as well as the condition, because the
            # only way a real kernel gets here is a *late*
            # ``pack_untilize_dest_init``, and the condition alone reads like a
            # missing tt-sim feature when it is a missing config write. See
            # docs/cost-model-caveats-for-consumers.md, "pack_untilize_dest on
            # Blackhole", for the measured trace.
            raise NotImplementedError(
                "PACR DST_ACCESS_STRIDED_MODE with DEST_ACCESS_CFG remap_addrs="
                f"{int(dst.dest_remap_addrs)} swizzle_32b={int(dst.dest_swizzle_32b)} "
                "is not modelled; the stride-of-16 Dst read is only defined with "
                "both set, and nothing specifies what it would be otherwise "
                "(ttsim refuses the same instruction: 'we currently require "
                "strided mode to be tied to the swizzle_32b and remap_addrs "
                "features'). This is reached by a kernel calling "
                "pack_untilize_dest_init AFTER its math rather than before: on "
                "Blackhole that init runs MATH(_llk_math_reconfig_remap_), which "
                "spins on the MATH_PACK semaphore until the pack it is racing has "
                "finished, so a PACK thread already past tile_regs_wait() issues "
                "this PACR before MATH ever writes the bits. Move "
                "pack_untilize_dest_init ahead of the math"
            )
        if packMask not in (0, 1, 3):
            raise NotImplementedError(
                f"PACR DST_ACCESS_STRIDED_MODE with read_intf_sel={packMask:#06b} is "
                "not modelled; only the contiguous interface selections (all four, "
                "one, or the low pair) have a reference behaviour"
            )
        return 1

    def getIPackerConfig(self, i, s, one_id, two_id=0):
        if s == 2:
            match i:
                case 0:
                    return "THCON_SEC0_REG" + str(one_id) + "_"
                case 1:
                    return "THCON_SEC0_REG" + str(one_id) + "_"
                case 2:
                    return "THCON_SEC1_REG" + str(one_id) + "_"
                case 3:
                    return "THCON_SEC1_REG" + str(one_id) + "_"
                case _:
                    raise ValueError()
        elif s == 4:
            match i:
                case 0:
                    return "THCON_SEC0_REG" + str(one_id) + "_"
                case 1:
                    return "THCON_SEC0_REG" + str(two_id) + "_"
                case 2:
                    return "THCON_SEC1_REG" + str(one_id) + "_"
                case 3:
                    return "THCON_SEC1_REG" + str(two_id) + "_"
                case _:
                    raise ValueError()

    def generate_input_address(
        self, stateID, issue_thread, flush, zeroWrite, addrMod, packMask, ovrdThreadId
    ):
        ADCsToAdvance = [False, False, False]
        for i in self.participating_packers(packMask):
            whichADC = issue_thread
            if ovrdThreadId:
                whichADC = self.getConfigValue(
                    stateID,
                    self.getConfigValue(
                        stateID, self.getIPackerConfig(i, 4, 1, 8) + "Addr_cnt_context"
                    ),
                )
                if whichADC == 3:
                    whichADC = 0

            adc = self.backend.getADC(whichADC).Packers.Channel[0]
            addr = (
                self.getConfigValue(stateID, "PCK0_ADDR_BASE_REG_0_Base")
                + adc.X
                * (self.getConfigValue(stateID, "PCK0_ADDR_BASE_REG_0_Base") & 0xF)
                + adc.Y
                * self.getConfigValue(stateID, "PCK0_ADDR_CTRL_XY_REG_0_Ystride")
                + adc.Z
                * self.getConfigValue(stateID, "PCK0_ADDR_CTRL_ZW_REG_0_Zstride")
                + adc.W
                * self.getConfigValue(stateID, "PCK0_ADDR_CTRL_ZW_REG_0_Wstride")
            )

            ADCsToAdvance[whichADC] = True

            in_df = self.getConfigValue(
                stateID, self.getIPackerConfig(i, 4, 1, 8) + "In_data_format"
            )

            self.packerI[i].inDataFormat = DataFormat(in_df)

            # How wide a Dst read is. A single global register (not per-packer
            # section), zero out of reset, and the only thing that decides it on
            # hardware -- ttsim reads Dst through `read_dst32b` when it is set
            # and `read_dst16b` when it is not, whatever the data formats say.
            # tt-metal's `_llk_pack_hw_configure_` /
            # `reconfig_packer_data_format` set it to
            # `is_32b_format || is_fp32_dest_acc_en`, so it is *implied* by a
            # 32-bit In_data_format but not equivalent to it: with
            # `fp32_dest_acc_en` over 16-bit circular buffers the pack source
            # format is 16-bit while Dst is still read 32 bits at a time.
            self.packerI[i].readDst32b = bool(
                self.getConfigValue(stateID, "PCK_DEST_RD_CTRL_Read_32b_data")
            )

            match in_df & 3:
                case 0:
                    # FP32, TF32, I32
                    bytesPerDatum = 4
                    ADC_X_Mask = 0x3
                case 1:
                    # FP16, BF16, I16
                    bytesPerDatum = 2
                    ADC_X_Mask = 0x7
                case _:
                    # All other formats
                    bytesPerDatum = 1
                    ADC_X_Mask = 0xF

            if flush:
                self.packerI[i].inputNumDatums = 0
            else:
                numDatums = (
                    self.backend.getADC(whichADC).Packers.Channel[1].X - adc.X + 1
                )
                if self.backend.blackhole:
                    # Blackhole PACR reinterprets the Wormhole `PackSel` bits
                    # (11:8) as `read_intf_sel`, selecting which of the four Dst
                    # read interfaces are active. With all four enabled (field
                    # 0) one PACR streams 4x the datums across four contiguous
                    # Dst rows (row = base + i//16); otherwise it scales by the
                    # number of selected interfaces. See ttsim TT_ARCH_VERSION==1
                    # pack path and the ISA PACR read_intf_sel field.
                    numDatums *= len(self.dst_read_interfaces(packMask))
                self.packerI[i].inputNumDatums = numDatums

            if zeroWrite or flush:
                self.packerI[i].inputSource = None
                self.packerI[i].inputSourceAddr = 0
                self.packerI[i].inputSourceStride = 0
            elif (
                i == 0
                and self.getConfigValue(
                    stateID,
                    self.getIPackerConfig(i, 4, 1, 8) + "Source_interface_selection",
                )
                == 1
            ):
                # Only low 18 bits of Addr used; high bits of L1 address come from L1_source_addr
                addr = (
                    self.getConfigValue(
                        stateID, self.getIPackerConfig(i, 4, 1, 8) + "L1_source_addr"
                    )
                    << 18
                ) + (addr & 0x3FFFF)
                addr = (addr & ~0xF) + bytesPerDatum * (adc.X & ADC_X_Mask)
                self.packerI[i].inputSource = PackerUnit.InputSource.L1
                self.packerI[i].inputSourceAddr = (
                    addr & 0x1FFFFF
                )  # Byte address into L1
                self.packerI[
                    i
                ].inputSourceStride = bytesPerDatum  # L1 is addressed in bytes
            else:
                addr = (int(addr / bytesPerDatum) & ~ADC_X_Mask) + (adc.X & ADC_X_Mask)
                offset = (
                    self.getConfigValue(
                        stateID, "DEST_TARGET_REG_CFG_PACK_SEC" + str(i) + "_Offset"
                    )
                    << 4
                )
                if i == 0 and offset == 32:
                    # This is a bug fix, a strange issue where unpacker core writes 32 after unpack
                    # into GPR 4, this is expected to hold the offset but is overwritten by this
                    # The other way of looking at it is that for packer 0 we should align with a
                    # segment, therefore if do not (% 64) then round down
                    offset = 0
                addr += offset
                self.packerI[i].inputSource = PackerUnit.InputSource.DST
                self.packerI[i].inputSourceAddr = (
                    addr & 0x3FFF
                )  # Datum index into Dst; `>> 4` is row, `& 0xf` is column
                self.packerI[
                    i
                ].inputSourceStride = 1  # Dst is addressed in datums, not in bytes

        for i in range(3):
            if not ADCsToAdvance[i]:
                continue

            adc = self.backend.getADC(i).Packers.Channel[0]

            AM_KEY = "ADDR_MOD_PACK_SEC" + str(addrMod)

            if self.backend.getThreadConfigValue(issue_thread, AM_KEY + "_YsrcClear"):
                adc.Y = 0
                adc.Y_Cr = 0
            elif self.backend.getThreadConfigValue(issue_thread, AM_KEY + "_YsrcCR"):
                adc.Y_Cr += self.backend.getThreadConfigValue(
                    issue_thread, AM_KEY + "_YsrcIncr"
                )
                adc.Y = adc.Y_Cr
            else:
                incr = self.backend.getThreadConfigValue(
                    issue_thread, AM_KEY + "_YsrcIncr"
                )
                # ttsim advances the pack Y counter (ch_y = ch_y + incr) in this
                # non-CR/non-clear branch, on *both* architectures — the update
                # sits outside every `#if TT_ARCH_VERSION` in its
                # TENSIX_EXECUTE_PACR. Setting it instead of adding pinned the
                # Dst read row across the successive PACRs that pack one
                # multi-face tile, so only the first face landed correctly:
                # `six`'s 32x32 output was the first full-tile pack to hit it
                # (four/five pack a single face, where Y is always 0 here so set
                # == accumulate) and a row-major `pack_untilize_dest`, which
                # interleaves the four faces into output rows, landed only the
                # first row. This was Blackhole-guarded when it was found, to
                # keep Wormhole byte-identical; the guard is gone because ttsim
                # does not have one and optests/untilize pins the behaviour on
                # both arches.
                adc.Y = (adc.Y + incr) & 0x1FFF  # ADC_Y_MASK (13 bits)

            if self.backend.getThreadConfigValue(issue_thread, AM_KEY + "_ZsrcClear"):
                adc.Z = 0
                adc.Z_Cr = 0
            else:
                z_incr = self.backend.getThreadConfigValue(
                    issue_thread, AM_KEY + "_ZsrcIncr"
                )
                # As with Y above, and equally arch-independent in ttsim:
                # setting the pack Z (face) counter dropped the face selection
                # between the PACRs of a multi-face tile, so every face but the
                # first read the wrong Dst rows.
                adc.Z = (adc.Z + z_incr) & 0xFF  # ADC_Z_MASK (8 bits)

    def generate_output_address(
        self, stateID, issue_thread, packMask, flush, last, addrMod, ovrdThreadId
    ):
        ADCsToAdvance = [False, False, False]
        active = self.participating_packers(packMask)
        for i in range(4):
            # Per the ISA OutputAddressGenerator.md pseudocode:
            #   Addr = L1_Dest_addr + !Sub_l1_tile_header_size
            # Sub_l1_tile_header_size is a 1-bit field; the +1 (when it is
            # 0) is what skips the per-tile header.
            pfx = self.getIPackerConfig(i, 4, 1, 8)
            addr = self.getConfigValue(stateID, pfx + "L1_Dest_addr") + (
                1 - self.getConfigValue(stateID, pfx + "Sub_l1_tile_header_size")
            )

            if i == 0:
                packer0InitialAddr = addr
            elif get_nth_bit(packer0InitialAddr, 31):
                addr += packer0InitialAddr

            # Only a packer this PACR actually drives has its configuration
            # consulted; see participating_packers. The addresses above are
            # computed for all four because packer 0's base chains into the
            # others' on Wormhole.
            if i not in active:
                continue

            whichADC = issue_thread
            if ovrdThreadId:
                whichADC = self.getConfigValue(
                    stateID, self.getIPackerConfig(i, 4, 1, 8) + "Addr_cnt_context"
                )
                if whichADC == 3:
                    whichADC = 0

            adc = self.backend.getADC(whichADC).Packers.Channel[1]
            yzw_addr = (
                self.getConfigValue(stateID, "PCK0_ADDR_BASE_REG_1_Base")
                + adc.Y
                * self.getConfigValue(stateID, "PCK0_ADDR_CTRL_XY_REG_1_Ystride")
                + adc.Z
                * self.getConfigValue(stateID, "PCK0_ADDR_CTRL_ZW_REG_1_Zstride")
                + adc.W
                * self.getConfigValue(stateID, "PCK0_ADDR_CTRL_ZW_REG_1_Wstride")
            )
            addr += yzw_addr & ~0xF
            ADCsToAdvance[whichADC] = True

            if self.getConfigValue(
                stateID, self.getIPackerConfig(i, 4, 1, 8) + "Add_l1_dest_addr_offset"
            ):
                raise NotImplementedError()

            if (
                addr
                > self.getConfigValue(
                    stateID, self.getIPackerConfig(i, 2, 2) + "Unpack_limit_address"
                )
                * 2
                + 1
            ):
                addr -= (
                    self.getConfigValue(
                        stateID, self.getIPackerConfig(i, 2, 2) + "Unpack_fifo_size"
                    )
                    * 2
                )

            performingCompression = not (
                get_nth_bit(
                    self.getConfigValue(
                        stateID,
                        self.getIPackerConfig(i, 2, 1)
                        + "All_pack_disable_zero_compress",
                    ),
                    i,
                )
                if self.getConfigValue(
                    stateID,
                    self.getIPackerConfig(i, 2, 1)
                    + "All_pack_disable_zero_compress_ovrd",
                )
                else self.getConfigValue(
                    stateID, self.getIPackerConfig(i, 2, 1) + "Disable_zero_compress"
                )
            )
            if performingCompression:
                # Pack-side zero compression is not modelled. The field is
                # inverted (`Disable_zero_compress`), so a section that was
                # never written reads as "compressing" -- which is why this
                # must only ever be asked of a packer that this PACR drives
                # (see participating_packers); tt-metal sets the bit for every
                # packer it does configure.
                raise NotImplementedError(
                    f"Packer {i} requests zero compression, which is not modelled"
                )

            outDataFormat = self.getConfigValue(
                stateID, self.getIPackerConfig(i, 4, 1, 8) + "Out_data_format"
            )

            if outDataFormat & 2:
                raise NotImplementedError(
                    f"Packer {i} output data format {outDataFormat} is under 16 "
                    "bits per datum, which is not modelled"
                )

            self.packerI[i].outDataFormat = DataFormat(outDataFormat)
            self.packerI[i].outBytes = ceil(
                DATA_FORMAT_TO_BITS[self.packerI[i].outDataFormat] / 8
            )

            if self.packerI[i].datastreamNeedsNewAddr:
                self.packerI[i].byteAddress = (addr & 0x1FFFF) << 4
                self.packerI[i].datastreamNeedsNewAddr = False
                # The tile-pack-generator counters are per datastream: they only
                # advance while a datastream is open and restart at zero on the
                # first PACR of the next one (ttsim's `packer_valid ? tpg : 0`).
                self.packerI[i].tpgX = 0
                self.packerI[i].tpgY = 0

            if last or flush:
                self.packerI[i].datastreamNeedsNewAddr = True

        for i in range(3):
            if not ADCsToAdvance[i]:
                continue

            adc = self.backend.getADC(i).Packers.Channel[1]

            AM_KEY = "ADDR_MOD_PACK_SEC" + str(addrMod)

            # Channel 1 is the *output* counter, so it advances on the AddrMod's
            # Ydst/Zdst fields — Ysrc/Zsrc belong to channel 0 and are applied by
            # generate_input_address. Reading the src fields here coupled the two
            # counters: `pack_untilize_dest` programs YsrcIncr = 15 to walk Dst
            # rows and leaves Ydst at 0, so the output address counter was being
            # dragged 15 rows per PACR. See Packers/OutputAddressGenerator.md.
            if self.backend.getThreadConfigValue(issue_thread, AM_KEY + "_YdstClear"):
                adc.Y = 0
                adc.Y_Cr = 0
            elif self.backend.getThreadConfigValue(issue_thread, AM_KEY + "_YdstCR"):
                adc.Y_Cr = (
                    adc.Y_Cr
                    + self.backend.getThreadConfigValue(
                        issue_thread, AM_KEY + "_YdstIncr"
                    )
                ) & 0x1FFF
                adc.Y = adc.Y_Cr
            else:
                incr = self.backend.getThreadConfigValue(
                    issue_thread, AM_KEY + "_YdstIncr"
                )
                adc.Y = (adc.Y + incr) & 0x1FFF  # ADC_Y_MASK (13 bits)

            if self.backend.getThreadConfigValue(issue_thread, AM_KEY + "_ZdstClear"):
                adc.Z = 0
                adc.Z_Cr = 0
            else:
                z_incr = self.backend.getThreadConfigValue(
                    issue_thread, AM_KEY + "_ZdstIncr"
                )
                adc.Z = (adc.Z + z_incr) & 0xFF  # ADC_Z_MASK (8 bits)

    def edge_masks_for_pacr(self, stateID, packer):
        """Resolve packer ``packer``'s per-tile-row edge masks for this PACR.

        Every datum the packer emits is gated by a 16-bit mask, one bit per
        column: a zero bit writes a zero datum instead of reading Dst. Which
        mask applies is chosen per tile row: ``PCK_EDGE_TILE_ROW_SET_SELECT``
        picks one of the two ``TILE_ROW_SET_MAPPING`` tables for this packer,
        and that table maps the current tile-pack-generator row to
        ``PCK_EDGE_OFFSET_SEC0_mask`` or ``PCK_EDGE_OFFSET_SEC1_mask``.

        The reduce LLK drives exactly this to zero everything but the reduced
        datums (``_llk_pack_reduce_mask_config_``): a whole column for a row
        reduce, the first row for a column reduce, a single datum for a scalar
        reduce. Without it a reduce output carries whatever else was left in
        Dst. The default configuration (mapping 0, SEC0 = 0xFFFF) passes every
        datum through, so this is inert outside reduce.

        Returns ``(masks, readsPerPlane)`` where ``masks[row]`` is the 16-bit
        mask for tile-pack-generator row ``row`` and ``readsPerPlane`` is the
        row count the generator wraps at.
        """
        if self.getConfigValue(stateID, "PCK_EDGE_MODE_mode"):
            raise NotImplementedError("Packer per-datum edge mode is not modelled")

        rowSetSelect = (
            self.getConfigValue(stateID, "PCK_EDGE_TILE_ROW_SET_SELECT_select")
            >> (2 * packer)
        ) & 3
        sectionMask = [
            self.getConfigValue(stateID, f"PCK_EDGE_OFFSET_SEC{s}_mask")
            for s in range(4)
        ]
        masks = [
            sectionMask[
                self.getConfigValue(
                    stateID,
                    f"TILE_ROW_SET_MAPPING_{rowSetSelect}_row_set_mapping_{row}",
                )
            ]
            for row in range(16)
        ]
        readsPerPlane = self.getConfigValue(
            stateID, "PACK_COUNTERS_SEC0_pack_reads_per_xy_plane"
        )
        return masks, max(1, min(16, readsPerPlane))

    def handle_pacr(self, instruction_info, issue_thread, instr_args):
        self._check_unmodelled_bh_fields(instruction_info)
        last = instr_args["Last"]
        flush = get_nth_bit(instr_args["Flush"], 0)
        ovrdThreadId = instr_args["OvrdThreadId"]
        packMask = instr_args["PackSel"]
        zeroWrite = get_nth_bit(instr_args["ZeroWrite"], 0)
        addrMod = self._read_addr_mode(instruction_info, instr_args)
        strided = self._read_dst_access_mode(instruction_info, packMask)

        stateID = self.backend.getThreadConfigValue(
            issue_thread, "CFG_STATE_ID_StateID"
        )

        self.generate_input_address(
            stateID, issue_thread, flush, zeroWrite, addrMod, packMask, ovrdThreadId
        )
        self.generate_output_address(
            stateID, issue_thread, packMask, flush, last, addrMod, ovrdThreadId
        )

        # Blackhole's single packer gathers its datums through the Dst read
        # interfaces `read_intf_sel` picks: 16 datums per interface, interface
        # `k` reading Dst row `base + k`. ``None`` on Wormhole, where each
        # packer reads its rows contiguously.
        readInterfaces = (
            self.dst_read_interfaces(packMask) if self.backend.blackhole else None
        )

        for i in self.participating_packers(packMask):
            addr = self.packerI[i].byteAddress

            row_start = int(self.packerI[i].inputSourceAddr / (16))
            rows = int(self.packerI[i].inputNumDatums / 16)

            if self.getDiagnosticSettings().reportPacking():
                print(
                    f"Packer {i}: copy from dst at {self.packerI[i].inputSourceAddr} (row start "
                    f"= {row_start}, num rows= {rows}) total size="
                    f"{self.packerI[i].inputNumDatums}, to L1 {hex(addr)}, read data type "
                    f"{DATA_FORMAT_TO_NAME[self.packerI[i].inDataFormat]} -> write data type "
                    f"{DATA_FORMAT_TO_NAME[self.packerI[i].outDataFormat]}"
                )

            edgeMasks, readsPerPlane = self.edge_masks_for_pacr(stateID, i)

            # For example four need an extra two for alignment also
            for j in range(self.packerI[i].inputNumDatums):
                idx = self.packerI[i].inputSourceAddr + j
                row = idx >> 4
                col = idx & 0xF
                if strided:
                    # DST_ACCESS_STRIDED_MODE: the interfaces step a whole
                    # 16-row face apart rather than a row, so run `n` of this
                    # PACR's datums comes from Dst row `base + 16*n`. See
                    # _read_dst_access_mode.
                    row = row_start + self.STRIDED_DST_ROW_STRIDE * (j >> 4)
                elif readInterfaces is not None:
                    # Rebase the row on the interface supplying this 16-datum
                    # run: `readInterfaces` is [0,1,2,3] for the all-on
                    # selection (the contiguous walk above), [0,2] for
                    # `0b0101` and [1,3] for `0b1010`.
                    run = j >> 4
                    row = row_start + readInterfaces[run % len(readInterfaces)]

                masked = not get_nth_bit(edgeMasks[self.packerI[i].tpgY & 0xF], col)
                self.packerI[i].tpgX += 1
                if self.packerI[i].tpgX == 16:
                    self.packerI[i].tpgX = 0
                    self.packerI[i].tpgY += 1
                    if self.packerI[i].tpgY == readsPerPlane:
                        self.packerI[i].tpgY = 0

                if masked:
                    # Edge-masked out: a zero datum is emitted and Dst is not read
                    datum = 0
                else:
                    # The read width comes from PCK_DEST_RD_CTRL_Read_32b_data,
                    # never from the data format: `fp32_dest_acc_en` over 16-bit
                    # circular buffers leaves the pack source format 16-bit
                    # while Dst still has to be read 32 bits at a time. Reading
                    # 16 bits there returns only the high half of every other
                    # Dst row and leaves most of the tile unwritten.
                    if self.packerI[i].readDst32b:
                        raw_datum = self.backend.getDst().getDst32b(row, col)
                    else:
                        raw_datum = self.backend.getDst().getDst16b(row, col)

                    datum = self.formatConversion(
                        stateID,
                        self.packerI[i].inDataFormat,
                        self.packerI[i].outDataFormat,
                        raw_datum,
                        self.packerI[i].readDst32b,
                    )

                self.backend.addressable_memory.write(
                    addr, conv_to_bytes(datum, self.packerI[i].outBytes)
                )
                addr += self.packerI[i].outBytes

            # The output address carries forward across the PACRs that together
            # pack one tile: each PACR appends its datums after the previous
            # one, so the first PACR lands at the CB page base (what the
            # consumer reads) and later PACRs spill past it.
            # generate_output_address only recomputes byteAddress on a new
            # datastream (datastreamNeedsNewAddr, set on last/flush), so this
            # intra-tile advance is what the next PACR reads. Per
            # Packers/OutputAddressGenerator.md the datum buffers "persist
            # between consecutive pack instructions" and ByteAddress is
            # incremented as they drain — on both architectures, and ttsim
            # likewise stores packer_dst_addr back unconditionally. This was
            # Blackhole-guarded when it was found; without it a row-major
            # `pack_untilize_dest`, whose 16 PACRs each emit one half-row,
            # rewrote the same 16 datums sixteen times and only one output row
            # survived.
            self.packerI[i].byteAddress = addr

    def bf16ToOutFormat(self, bf16_data, outDataFormat):
        """The pack's late conversion out of a bf16 datum.

        Shared by the two ways one arrives at bf16: a 16-bit Dst read of a bf16
        Dst, and the fp32 narrowing a 32-bit Dst read does first.
        """
        match outDataFormat:
            case DataFormat.FP32:
                return bf16_data << 16
            case DataFormat.BF16:
                return bf16_data
            case DataFormat.FP16:
                return DataFormatConversions.FP32ToFP16(bf16_data << 16)
            case _:
                raise NotImplementedError()

    def formatConversion(
        self, stateID, inDataFormat, outDataFormat, raw_datum, read32b=False
    ):
        if read32b and DATA_FORMAT_TO_BITS[inDataFormat] != 32:
            # A 32-bit Dst read under a narrower pack source format: what
            # `fp32_dest_acc_en` over 16-bit circular buffers produces. Dst
            # holds fp32, so the packer narrows it to the source format first
            # (ttsim's `intermediate_format` step) and only then runs the usual
            # source -> output conversion below.
            if inDataFormat == DataFormat.BF16:
                return self.bf16ToOutFormat(
                    DataFormatConversions.FP32InDstToBF16(raw_datum), outDataFormat
                )
            raise NotImplementedError(
                "32-bit Dst read (PCK_DEST_RD_CTRL_Read_32b_data) with pack source "
                f"format {DATA_FORMAT_TO_NAME[inDataFormat]} is not modelled"
            )
        match inDataFormat:
            case DataFormat.FP32:
                fp32_data = DataFormatConversions.FP32InDstToFP32(raw_datum)
                match outDataFormat:
                    case DataFormat.FP32:
                        return fp32_data
                    case DataFormat.FP16:
                        return DataFormatConversions.FP32ToFP16(fp32_data)
                    case DataFormat.BF16:
                        return DataFormatConversions.FP32ToBF16(fp32_data)
                    case _:
                        raise NotImplementedError()
            case DataFormat.TF32:
                raise NotImplementedError()
            case DataFormat.BF16:
                return self.bf16ToOutFormat(
                    DataFormatConversions.BF16InDstToBF16(raw_datum), outDataFormat
                )
            case DataFormat.FP16:
                match outDataFormat:
                    case DataFormat.FP32:
                        return DataFormatConversions.FP16InDstToFP32(raw_datum)
                    case DataFormat.FP16:
                        return DataFormatConversions.FP16InDstToFP16(raw_datum)
                    case DataFormat.BF16:
                        return DataFormatConversions.FP16InDstToFP32(raw_datum) >> 16
                    case _:
                        raise NotImplementedError()
            case DataFormat.INT32:
                assert outDataFormat == DataFormat.INT32
                return raw_datum
            case DataFormat.INT16:
                assert outDataFormat == DataFormat.INT16
                return raw_datum
            case DataFormat.UINT8:
                assert outDataFormat == DataFormat.UINT8
                return raw_datum
            case _:
                raise NotImplementedError()
