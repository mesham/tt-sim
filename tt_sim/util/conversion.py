def conv_to_bytes(val, width=4, signed=False):
    if isinstance(val, int):
        return val.to_bytes(width, byteorder="little", signed=signed)
    elif isinstance(val, list):
        byte_data = bytearray()
        for el in val:
            byte_data.extend(conv_to_bytes(el, signed=signed))
        return bytes(byte_data)
    else:
        raise NotImplementedError()


def conv_to_int32(val, signed=True):
    if isinstance(val, bytes):
        return int.from_bytes(val, byteorder="little", signed=signed)
    else:
        raise NotImplementedError()


def conv_to_uint32(val):
    return conv_to_int32(val, False)


def insert_bytes(target: int, source: int, num_bytes: int, bit_position: int) -> int:
    """
    Inserts `num_bytes` bytes from `source` into `target` at `bit_position`.

    Args:
        target: The number to insert bytes into.
        source: The number to extract bytes from (lowest `num_bytes` bytes).
        num_bytes: Number of bytes to insert (each byte is 8 bits).
        bit_position: Starting bit position in `target` where bytes are inserted (0-based, from least significant bit).

    Returns:
        The modified target number with the bytes inserted.
    """
    # Calculate the number of bits to insert (num_bytes * 8)
    num_bits = num_bytes * 8

    # Create a mask for the bits to clear in the target
    mask = (1 << num_bits) - 1  # e.g., for 2 bytes: 0xFFFF (16 bits of 1s)
    mask = ~(mask << bit_position)  # Shift and invert to clear bits at bit_position

    # Clear the bits in the target at the specified position
    target = target & mask

    # Extract the lowest `num_bits` from the source and shift to the bit_position
    source_bits = source & ((1 << num_bits) - 1)  # Get lowest num_bits from source
    source_bits = source_bits << bit_position  # Shift to the correct position

    # Insert the source bits into the target
    result = target | source_bits

    return result


def get_nth_bit(value: int, n: int) -> int:
    if n < 0 or n > 31:
        raise ValueError("n must be between 0 and 31 for a 32-bit integer")
    return (value >> n) & 1


def clear_bit(value: int, position: int) -> int:
    if position < 0 or position > 31:
        raise ValueError("position must be between 0 and 31")

    mask = ~(1 << position)
    return value & mask


def set_bit(value: int, position: int) -> int:
    if position < 0 or position > 31:
        raise ValueError("Bit position must be in [0, 31]")
    return value | (1 << position)
