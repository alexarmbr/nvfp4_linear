#include <torch/extension.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <ATen/autocast_mode.h>
#include <pybind11/pybind11.h>
#include <cstdio>
#include <iostream>

// File containing the CUTLASS portion of the code.
#include "cutlass_bf16_gemm.hpp"

// Not strictly necessary, but here for convenience.
void cutlass_bf16_gemm(int M, int N, int K, half_t const* ptrA, half_t const* ptrB, float* ptrC, float const* ptrBias);

// This function is bound to "cutlass_gemm.mm". 
torch::Tensor cutlass_gemm(torch::Tensor A,  // A matrix (m x k)
                           torch::Tensor B,  // B matrix (k x n)
                           torch::Tensor out, // required out matrix (m x n)
                           torch::Tensor bias) {   // optional bias vector (m,)

  // Get problem dimensions
  const int M = A.sizes()[0];
  const int N = B.sizes()[1];
  const int K = A.sizes()[1];

  // Use the provided output tensor
  torch::Tensor C = out;

  // Check that all tensors are allocated on GPU device.
  if(!(A.device().is_cuda() && B.device().is_cuda() && C.device().is_cuda()))
    throw std::invalid_argument("cutlass_gemm only supports GPU device. Use .to(device=torch.device('cuda'))");

  // Check bias tensor if provided
  float const* bias_ptr = nullptr;
  if (bias.numel() > 0) {
    if (!bias.device().is_cuda())
      throw std::invalid_argument("bias tensor must be on GPU device");
    if (bias.sizes()[0] != N)
      throw std::invalid_argument("bias tensor must have size (N,) where N is the number of output columns");
    
    torch::Tensor _bias = bias.contiguous();
    bias_ptr = static_cast<float const*>(_bias.data_ptr());
  }

  // Ensuring that the matrices are contiguous. 
  torch::Tensor _A = A.contiguous();
  torch::Tensor _B = B.contiguous();
  torch::Tensor _C = C.contiguous();

  cutlass_bf16_gemm(M, N, K, 
                    static_cast<half_t const*>(_A.data_ptr()), 
                    static_cast<half_t const*>(_B.data_ptr()), 
                    static_cast<float*>(_C.data_ptr()),
                    bias_ptr);

  // If C was not contiguous, C != _C so copy the result back into C
  if(!C.is_contiguous())
    C.copy_(_C);

  // Return the Torch tensor back to PyTorch
  return C;
}

// Binding the function to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mm", &cutlass_gemm, pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("out"), pybind11::arg("bias"));
}
