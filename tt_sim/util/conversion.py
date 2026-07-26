import struct


def conv_to_bytes(val, width=4, signed=False):
    if isinstance(val, int):
        return val.to_bytes(width, byteorder="little", signed=signed)
    elif isinstance(val, float):
        return struct.pack("f", val)
    elif isinstance(val, list):
        byte_data = bytearray()
        for el in val:
            byte_data.extend(conv_to_bytes(el, width, signed=signed))
        return bytes(byte_data)
    elif isinstance(val, bytes):
        return val
    else:
        raise NotImplementedError()


def conv_to_int32(val, signed=True):
    if isinstance(val, bytes):
        return int.from_bytes(val, byteorder="little", signed=signed)
    elif isinstance(val, int):
        # Normalise to 32 bits so out-of-range ints (e.g. negative values held
        # in an lreg, or results wider than 32 bits) are interpreted the same
        # way the hardware would rather than passed through unchanged.
        val &= 0xFFFFFFFF
        if signed and val >= 0x80000000:
            val -= 0x100000000
        return val
    else:
        raise NotImplementedError()


def conv_to_uint32(val):
    if isinstance(val, float):
        return struct.unpack("I", conv_to_bytes(val))[0]
    else:
        return conv_to_int32(val, False)


def conv_to_float(val):
    if isinstance(val, int):
        return struct.unpack("f", conv_to_bytes(val))[0]
    elif isinstance(val, bytes):
        return struct.unpack("f", val)[0]
    else:
        raise ValueError()
