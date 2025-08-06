from collections.abc import Iterable
from lib.types import FractionalList


def create_fractional_list(percentages: list[float]) -> FractionalList | None:
    """Create a FractionalList from a list of percentages.

    Args:
        percentages (list[float]): A list of percentages that sum to 1.0.

    Returns:
        FractionalList | None: A FractionalList if the input is valid, otherwise None.
    """
    if not percentages or abs(sum(percentages) - 1.0) > 1e-6:
        return None
    return FractionalList(percentages)


def subset_n(n: int, percentages: FractionalList) -> list[int]:
    """Given a number n and a list of percentages, return a list of integers
    that represent the subset sizes of n according to the percentages.

    Args:
        n (int): The total number to be divided into subsets.
        percentages (list[float]): A list of percentages that sum to 1.0.

    Returns:
        list[int]: A list of integers representing the subset sizes.
    """
    subset_sizes = [int(n * p) for p in percentages]
    total_assigned = sum(subset_sizes)
    remainder = n - total_assigned

    for i in range(remainder):
        subset_sizes[i % len(subset_sizes)] += 1

    return subset_sizes
