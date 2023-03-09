from torch import BoolTensor, FloatTensor
import torch

def mask_to_bias(mask: BoolTensor, dtype: torch.dtype) -> FloatTensor:
    bias: FloatTensor = (1 - mask.to(dtype=dtype)) * -10000.0
    return bias
    