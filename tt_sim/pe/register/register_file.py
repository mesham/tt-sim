class RegisterFile:
    def __init__(self, registers, register_name_mapping):
        self.registers = registers
        self.register_name_mapping = register_name_mapping
        # Last-write recording for tracing. Callers clear before
        # executing an instruction and read after to learn what (if
        # anything) the instruction wrote. -1 = no write this window.
        self.last_write_idx: int = -1
        self.last_write_value: bytes = b""
        # Install a wrapper on each Register's write so the recording
        # happens transparently; ISA code keeps using
        # `register_file[idx].write(bytes)` unchanged.
        for idx, reg in enumerate(registers):
            self._install_write_hook(reg, idx)

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
        if isinstance(idx, int):
            assert idx < len(self.registers)
            return self.registers[idx]
        elif isinstance(idx, str):
            assert idx in self.register_name_mapping.keys()
            return self.registers[self.register_name_mapping[idx]]
        else:
            raise IndexError(
                f"Index of type '{type(idx)}' can not be used as a register lookup"
            )

    def __getitem__(self, idx):
        return self.get(idx)
