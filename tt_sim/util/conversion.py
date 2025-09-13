def conv_to_bytes(val, width=4, signed=False):
    if isinstance(val, int):
        return val.to_bytes(width, byteorder="little", signed=signed)
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
    else:
        raise NotImplementedError()


def conv_to_uint32(val):
    return conv_to_int32(val, False)
