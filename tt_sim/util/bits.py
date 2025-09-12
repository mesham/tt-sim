def replace_bits(target: int, value: int, position: int, n: int) -> int:
    """
    Take the lowest 'n' bits from 'value' and set them into 'x' starting at bit position 'p'.

    Args:
        x (int): Target integer whose bits are to be modified.
        value (int): Source of bits (we use its lowest 'n' bits).
        p (int): Starting bit position in 'x' (0 = LSB).
        n (int): Number of bits to take from 'value' and insert into 'x'.

    Returns:
        int: The modified integer.
    """
    # Step 1: Create a mask to clear n bits at position p in x
    mask = ((1 << n) - 1) << position
    target_cleared = target & ~mask

    # Step 2: Take exactly n LSBs from value and shift them into position p
    value_masked = (value & ((1 << n) - 1)) << position

    # Step 3: Combine cleared target and new bits
    return target_cleared | value_masked


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


def extract_bits(val, n, p):
    """
    Extract n bits from integer x starting at position p.

    Parameters:
    - x (int): The input integer.
    - p (int): The position to start extracting bits (0 = least significant bit).
    - n (int): The number of bits to extract.

    Returns:
    - int: The extracted bits as a new integer.
    """
    return (val >> p) & ((1 << n) - 1)
