#!/usr/bin/env python3
"""
Test script for bf16_to_fp4 extension.
Run this after building the extension with: python setup_bf16_to_fp4.py build_ext --inplace
"""

import torch
import numpy as np

from cutlass_gemm import bf16_to_fp4

# M, N = 8192 * 2, 8192 * 2

# def benchmark_copy(copy_func):
#     for i in range(5):
#         copy_func()
#     start_event = torch.cuda.Event(enable_timing=True)
#     end_event = torch.cuda.Event(enable_timing=True)
#     start_event.record()
#     for i in range(20):
#         copy_func()
#     end_event.record()
#     torch.cuda.synchronize()
#     elapsed_ms = start_event.elapsed_time(end_event) / 20
#     gibytes_moved = (M * N * 2 * 2) / (1024**3)
#     print(f"effective bandwidth: {gibytes_moved / (elapsed_ms / 1e3)} GiB/s")


# input_tensor = torch.randn(M, N, dtype=torch.bfloat16, device=torch.device('cuda'))

# my_copy_func = lambda: bf16_to_fp4(input_tensor)
# torch_copy_func = lambda: input_tensor.clone()

# print("bf16_to_fp4")
# benchmark_copy(my_copy_func)
# print("torch.clone")
# benchmark_copy(torch_copy_func)


small_input = torch.randn(8, 256, dtype=torch.bfloat16, device=torch.device('cuda'))
out, scale = bf16_to_fp4(small_input)
print(scale)






        

