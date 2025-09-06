def conv_to_bytes(val):
    if isinstance(val, int):
        return val.to_bytes(4, byteorder="little", signed=True)
    elif isinstance(val, list):
        byte_data = bytearray()
        for el in val:
            byte_data.extend(conv_to_bytes(el))
        return bytes(byte_data)
    else:
        print(type(val))
        raise NotImplementedError()


def conv_to_int32(val):
    if isinstance(val, bytes):
        return int.from_bytes(val, byteorder="little", signed=True)
    else:
        raise NotImplementedError()
