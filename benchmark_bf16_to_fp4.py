import torch
from cutlass_gemm import bf16_to_fp4

M, N = 8192 * 2, 8192 * 2

def benchmark_copy(copy_func):
    for i in range(5):
        copy_func()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for i in range(20):
        copy_func()
    end_event.record()
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event) / 20
    return elapsed_ms


input_tensor = torch.randn(M, N, dtype=torch.bfloat16, device=torch.device('cuda'))
my_copy_func = lambda: bf16_to_fp4(input_tensor)
torch_copy_func = lambda: input_tensor.clone()

print("bf16_to_fp4")
elapsed_ms = benchmark_copy(my_copy_func)
bytes_read = M * N * 2
bytes_written = (M * N / 16) + (M * N / 2)
gibytes_moved = (bytes_read + bytes_written) / (1024**3)
print(f"effective bandwidth: {gibytes_moved / (elapsed_ms / 1e3)} GiB/s")
print(f"elapsed_ms: {elapsed_ms}")

print("torch.clone")
elapsed_ms = benchmark_copy(torch_copy_func)
bytes_read = M * N * 2
bytes_written = M * N * 2
gibytes_moved = (bytes_read + bytes_written) / (1024**3)
print(f"effective bandwidth: {gibytes_moved / (elapsed_ms / 1e3)} GiB/s")
print(f"elapsed_ms: {elapsed_ms}")

# bf16_to_fp4 achieves 1541 GB/s, seems to be mostly memory bound by the writing on the fp4 values (vectorized to int32) back to gmem
# could probably be made faster by writing back to shared memory, and then using 128B writes to gmem