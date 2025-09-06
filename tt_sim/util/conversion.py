def conv_to_bytes(val, signed=False):
    if isinstance(val, int):
        return val.to_bytes(4, byteorder="little", signed=signed)
    elif isinstance(val, list):
        byte_data = bytearray()
        for el in val:
            byte_data.extend(conv_to_bytes(el, signed=signed))
        return bytes(byte_data)
    else:
        print(type(val))
        raise NotImplementedError()


def conv_to_int32(val, signed=True):
    if isinstance(val, bytes):
        return int.from_bytes(val, byteorder="little", signed=signed)
    else:
        raise NotImplementedError()


def conv_to_uint32(val):
    return conv_to_int32(val, False)
