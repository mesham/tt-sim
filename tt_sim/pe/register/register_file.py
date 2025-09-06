class RegisterFile:
    def __init__(self, registers, register_name_mapping):
        self.registers = registers
        self.register_name_mapping = register_name_mapping

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
