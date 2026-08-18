class RegisterFile:
    #: The owning core's :class:`~tt_sim.pe.rv.isa.zicsr_isa.CSRFile`, or ``None``
    #: when it has no CSRs. Held here as well as on the core because the ISA
    #: executors are handed the register file and nothing else, and ``fcsr``
    #: already lives in this file — the CSR file aliases that entry rather than
    #: keeping a second copy. A class attribute so nothing that builds a
    #: register file has to know about CSRs.
    csrs = None

    def __init__(self, registers, register_name_mapping):
        self.registers = registers
        self.register_name_mapping = register_name_mapping
        # Last-write recording for tracing. Callers clear before
        # executing an instruction and read after to learn what (if
        # anything) the instruction wrote. -1 = no write this window.
        self.last_write_idx: int = -1
        self.last_write_value: bytes = b""
        # Recording is off until someone asks for it: the hook is a Python
        # closure wrapped around Register.write, which is on the simulator's
        # hottest path, so it is not paid for when nothing is tracing.
        self.write_recording: bool = False

    def set_write_recording(self, enabled):
        """Install/remove the per-Register write hooks used by tracing.

        With the hooks installed, ``register_file[idx].write(bytes)`` records
        what it wrote; ISA code is unchanged either way.
        """
        if enabled == self.write_recording:
            return
        for idx, reg in enumerate(self.registers):
            if enabled:
                self._install_write_hook(reg, idx)
            else:
                # Drop the instance attribute, exposing Register.write again.
                del reg.write
        self.write_recording = enabled
        self.clear_write_record()

    def _install_write_hook(self, reg, idx):
        original_write = reg.write

        def wrapped_write(value):
            original_write(value)
            self.last_write_idx = idx
            self.last_write_value = value

        reg.write = wrapped_write

    def clear_write_record(self):
        self.last_write_idx = -1
        self.last_write_value = b""

    def get(self, idx):
        # The most-called function in the simulator: take the integer index
        # straight to the list and only consult the name map when that is not
        # an index (i.e. a register name), rather than type-testing up front.
        try:
            return self.registers[idx]
        except TypeError:
            pass
        try:
            return self.registers[self.register_name_mapping[idx]]
        except (KeyError, TypeError):
            raise IndexError(
                f"Index of type '{type(idx)}' can not be used as a register lookup"
            ) from None

    __getitem__ = get
