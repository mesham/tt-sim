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
    - n (int): The number of bits to extract.
    - p (int): The position to start extracting bits (0 = least significant bit).

    Returns:
    - int: The extracted bits as a new integer.
    """
    return (val >> p) & ((1 << n) - 1)


def get_bits(value: int, start: int, end: int) -> int:
    """
    Extract bits from `start` to `end` (inclusive) from a signed 32-bit integer.

    Bit positions are 0-indexed from the LSB (rightmost bit).

    Example:
        extract_bits(0b111011, 1, 3) => 0b101 => 5

    Args:
        value (int): The 32-bit signed integer.
        start (int): Starting bit index (inclusive).
        end (int): Ending bit index (inclusive).

    Returns:
        int: The extracted bits as an integer.
    """
    if not (0 <= start <= 31 and 0 <= end <= 31):
        raise ValueError("start and end must be between 0 and 31")
    if start > end:
        raise ValueError("start must be less than or equal to end")

    # Ensure it's treated as unsigned 32-bit
    value &= 0xFFFFFFFF

    # Create bitmask
    num_bits = end - start + 1
    mask = (1 << num_bits) - 1

    # Shift and mask
    return (value >> start) & mask


def int_to_bin_list(value: int, width: int = 32) -> list[int]:
    """
    Convert an integer to a list of binary digits (0 or 1).

    Args:
        value (int): The integer to convert.
        width (int): The number of bits to include (default is 32).

    Returns:
        List[int]: List of binary digits, from MSB to LSB.
    """
    value &= (1 << width) - 1  # Ensure it's within the desired bit width
    return [(value >> i) & 1 for i in reversed(range(width))]
