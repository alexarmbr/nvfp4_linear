rm -f bf16_to_fp4.cubin
nvcc -cubin -arch=sm_100a bf16_to_fp4.cu -o bf16_to_fp4.cubin -I/root/cutlass_gemm/cutlass/include
cuobjdump --dump-sass bf16_to_fp4.cubin







