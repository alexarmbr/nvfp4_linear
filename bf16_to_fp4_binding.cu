#include <torch/extension.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <ATen/autocast_mode.h>
#include <pybind11/pybind11.h>
#include <cstdio>
#include <iostream>
#include <tuple>

// Macro to check CUDA memory allocation errors
#define CUDA_CHECK_MALLOC(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      throw std::runtime_error(std::string("CUDA malloc failed: ") + cudaGetErrorString(err) + \
                              " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
  } while(0)

// Forward declaration of the CUDA function
void bf16_to_fp4(cutlass::bfloat16_t const* in, cutlass::float_e2m1_t* data_out, cutlass::float_ue4m3_t* scale_out, int rows, int cols);

// Python wrapper function
torch::Tensor bf16_to_fp4_wrapper(torch::Tensor input) {
    
    // Check that input tensor is allocated on GPU device
    if (!input.device().is_cuda()) {
        throw std::invalid_argument("bf16_to_fp4 only supports GPU device. Use .to(device=torch.device('cuda'))");
    }
    
    // Check that input tensor is bfloat16
    if (input.dtype() != torch::kBFloat16) {
        throw std::invalid_argument("bf16_to_fp4 only supports bfloat16 input. Use .to(dtype=torch.bfloat16)");
    }
    
    // Check that input tensor is contiguous and 2D
    if (input.dim() != 2) {
        throw std::invalid_argument("Input tensor must be 2D (rows, cols)");
    }
    
    if (input.stride(1) != 1) {
        throw std::invalid_argument("Input tensor must be row major (contiguous)");
    }
    
    // Get dimensions
    constexpr int nvfp4_scale_vector_length = 16;
    const int rows = input.sizes()[0];
    const int data_cols = input.sizes()[1];
    assert(data_cols % nvfp4_scale_vector_length == 0);
    const int scale_cols = data_cols / nvfp4_scale_vector_length;

    
    // Ensure input is contiguous
    torch::Tensor _input = input.contiguous();
    
    // Create output tensor for FP4 data using float8_e2m1 format
    // torch::Tensor data_out = torch::empty({rows, data_cols}, 
    //                                      torch::dtype(torch::kFloat8_e2m1).device(input.device()));
    const unsigned int data_out_bytes = rows * data_cols / 2;
    cutlass::float_e2m1_t* data_out;
    CUDA_CHECK_MALLOC(cudaMalloc(&data_out, data_out_bytes));
    
    // Scale tensor using fp8 e4m3 format
    torch::Tensor scale_out = torch::empty({rows, scale_cols}, 
                                          torch::dtype(torch::kFloat8_e4m3fn).device(input.device()));
    
    // Call the CUDA kernel directly with PyTorch tensor data pointers
    bf16_to_fp4(static_cast<cutlass::bfloat16_t const*>(_input.data_ptr()),
                data_out,
                reinterpret_cast<cutlass::float_ue4m3_t*>(scale_out.data_ptr()),
                rows, data_cols);
    
    // Allocate CPU memory to copy GPU results back for printing
    cutlass::float_e2m1_t* data_out_cpu = (cutlass::float_e2m1_t*)malloc(data_out_bytes);
    
    // Copy GPU memory to CPU memory
    CUDA_CHECK_MALLOC(cudaMemcpy(data_out_cpu, data_out, data_out_bytes, cudaMemcpyDeviceToHost));
    
    // print the top left 16x16 of data_out using CPU memory
    // Note: each byte contains 2 FP4 values (4 bits each)
    uint8_t* raw_bytes = reinterpret_cast<uint8_t*>(data_out_cpu);
    int byte_stride = data_cols / 2;
    
    // for (int j = 0; j < 2; j++) {
    //     for (int i = 0; i < 16; i++) {
    //         uint8_t byte = raw_bytes[j * byte_stride + i];
    //         uint8_t nibble0 = byte & 0x0F;
    //         uint8_t nibble1 = (byte & 0xF0) >> 4;
    //         cutlass::float_e2m1_t result = cutlass::float_e2m1_t::bitcast(nibble0);
    //         cutlass::float_e2m1_t result2 = cutlass::float_e2m1_t::bitcast(nibble1);
    //         printf("%.2f %.2f ", float(result), float(result2));
    //     }
    //     printf("\n");
    // }

    // Free both GPU and CPU memory
    free(data_out_cpu);
    cudaFree(data_out);
    
    return scale_out;
}

// Binding the function to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bf16_to_fp4", &bf16_to_fp4_wrapper, 
          "Convert bfloat16 tensor to fp4 format with scaling factors",
          pybind11::arg("input"));
} 