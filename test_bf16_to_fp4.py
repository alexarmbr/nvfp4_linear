#!/usr/bin/env python3
"""
Test script for bf16_to_fp4 extension.
Run this after building the extension with: python setup_bf16_to_fp4.py build_ext --inplace
"""

import torch
import numpy as np

from cutlass_gemm import bf16_to_fp4


def compute_scales(input_tensor):
    scale_vector_length = 16
    assert input_tensor.shape[1] % scale_vector_length == 0
    input_tensor = input_tensor.view(input_tensor.shape[0], -1, scale_vector_length)
    max_values = torch.max(input_tensor, dim=2).values
    max_values = max_values.to(torch.float32)
    max_values = max_values / 6.0
    max_values = torch.clamp(max_values, max=448.0)
    max_values = max_values.to(torch.float8_e4m3fn)
    return max_values


# TODO verify with pencil and paper that that the quantization of 0,1,....15 is correct
# TODO scale factors should not be negative
small_input = torch.ones(8, 256, dtype=torch.bfloat16, device=torch.device('cuda')) * 6.0
for i in range(32):
    small_input[0, i] = i

for i in range(32):
    small_input[1, i] = (32 - i) * -1


scale = bf16_to_fp4(small_input)

scale_gt = compute_scales(small_input)
scale = scale.to(torch.float32)
scale_gt = scale_gt.to(torch.float32)
assert (scale == scale_gt).all()

print(scale)








        

