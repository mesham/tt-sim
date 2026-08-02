from enum import Enum


class RegisterAccessMode(Enum):
    R = 1
    RW = 2


class Register:
    """A fixed-width architectural register, stored as plain ``bytes``.

    This is the hottest data structure in the simulator — every RV32 GPR is one
    of these, and a single instruction reads/writes it several times — so the
    representation is deliberately the cheapest thing CPython offers. ``bytes``
    is immutable, so ``read()`` can hand out the stored object with no copy and
    ``write()`` is a single attribute store. (It used to be a 4-element numpy
    ``uint8`` array; numpy's per-call overhead on a 4-byte scalar was ~0.5 µs
    per read and ~2.2 µs per write, i.e. a large fraction of the whole
    interpreter's wall clock.)
    """

    def __init__(
        self,
        size,
        init_val=None,
        access_mode=RegisterAccessMode.RW,
        error_on_write_to_read=True,
    ):
        self.size = size
        self.access_mode = access_mode
        self.error_on_write_to_read = error_on_write_to_read

        if init_val is None:
            self.value = bytes(size)
        else:
            assert isinstance(init_val, bytes)
            self.value = self._fit(init_val, bytes(size))

    @staticmethod
    def _fit(value, current):
        """Coerce ``value`` to the register width, keeping ``current``'s tail."""
        size = len(current)
        if len(value) >= size:
            return value[:size]
        return value + current[len(value) :]

    def read(self):
        return self.value

    def read_uint(self):
        """Value as an unsigned integer — ``conv_to_uint32(reg.read())`` without
        the two helper calls, which the ISA modules make several times per
        simulated instruction."""
        return int.from_bytes(self.value, "little")

    def read_int(self):
        """Value as a signed integer; the signed counterpart of read_uint."""
        return int.from_bytes(self.value, "little", signed=True)

    def write(self, value):
        if self.access_mode != RegisterAccessMode.RW:
            # Some ISA might error on this, others such as RV simply ignore it
            if self.error_on_write_to_read:
                raise Exception("Can not call set on a read only register")
            else:
                return
        assert isinstance(value, bytes)
        self.value = value if len(value) == self.size else self._fit(value, self.value)
