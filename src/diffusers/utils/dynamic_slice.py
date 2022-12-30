from torch import Tensor
from typing import List

def dynamic_slice(
    x: Tensor,
    starts: List[int],
    sizes: List[int],
) -> Tensor:
    slicing = [slice(start, start + size) for start, size in zip(starts, sizes)]
    return x[slicing]