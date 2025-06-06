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
// void bf16_to_fp4(cutlass::bfloat16_t const* in, cutlass::float_e2m1_t* data_out, cutlass::float_ue4m3_t* scale_out, int rows, int cols);
void bf16_to_fp4(cutlass::bfloat16_t const* in, cutlass::bfloat16_t* data_out, cutlass::float_ue4m3_t* scale_out, int rows, int cols);


// Python wrapper function
std::tuple<torch::Tensor, torch::Tensor> bf16_to_fp4_wrapper(torch::Tensor input) {
    
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
    const int rows = input.sizes()[0];
    const int cols = input.sizes()[1];
    
    // Ensure input is contiguous
    torch::Tensor _input = input.contiguous();
    
    // Create output tensors
    // Note: Using uint8 for fp4 data since PyTorch doesn't have native fp4 support
    // Each fp4 value will be stored in the lower 4 bits of a uint8
    // torch::Tensor data_out = torch::empty({rows, cols}, 
    //                                      torch::dtype(torch::kUInt8).device(input.device()));
    torch::Tensor data_out = torch::empty_like(input);
    
    // Scale tensor using fp8 e4m3 format
    torch::Tensor scale_out = torch::empty({rows, cols}, 
                                          torch::dtype(torch::kFloat8_e4m3fn).device(input.device()));
    
    // Call the CUDA kernel directly with PyTorch tensor data pointers
    bf16_to_fp4(static_cast<cutlass::bfloat16_t const*>(_input.data_ptr()),
                // reinterpret_cast<cutlass::float_e2m1_t*>(data_out.data_ptr()),
                static_cast<cutlass::bfloat16_t*>(data_out.data_ptr()),
                reinterpret_cast<cutlass::float_ue4m3_t*>(scale_out.data_ptr()),
                rows, cols);
    
    return std::make_tuple(data_out, scale_out);
}

// Binding the function to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bf16_to_fp4", &bf16_to_fp4_wrapper, 
          "Convert bfloat16 tensor to fp4 format with scaling factors",
          pybind11::arg("input"));
} 