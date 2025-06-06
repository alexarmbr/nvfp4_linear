#include <torch/extension.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <ATen/autocast_mode.h>
#include <pybind11/pybind11.h>
#include <cstdio>
#include <iostream>

// Macro to check CUDA memory allocation errors
#define CUDA_CHECK_MALLOC(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      throw std::runtime_error(std::string("CUDA malloc failed: ") + cudaGetErrorString(err) + \
                              " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
  } while(0)


// File containing the CUTLASS portion of the code.
#include "cutlass_fp4_gemm.hpp"
#include "cutlass_fp16_gemm.hpp"


void cutlass_fp4_gemm(int M, int N, int K, cutlass::float_e2m1_t const* ptrA, cutlass::float_e2m1_t const* ptrB, cutlass::float_ue4m3_t const* ptrA_sf, cutlass::float_ue4m3_t const* ptrB_sf, cutlass::bfloat16_t* ptrC, float const* ptrBias);
void cutlass_fp16_gemm(int M, int N, int K, half_t const* ptrA, half_t const* ptrB, float* ptrC, float const* ptrBias);

void bf16_to_fp4(cutlass::bfloat16_t const* in, cutlass::float_e2m1_t* data_out, cutlass::float_ue4m3_t* scale_out, int rows, int cols);

torch::Tensor fp4_gemm(torch::Tensor A, torch::Tensor B, torch::Tensor bias) {
  
  if (A.sizes()[1] != B.sizes()[1]) {
    throw std::invalid_argument("cutlass_gemm only supports matrices with the same number of rows and columns. Use .t() to transpose the matrix");
  }

  // Get problem dimensions
  const int M = A.sizes()[0];
  const int N = B.sizes()[0];
  const int K = A.sizes()[1];

  // Check that all tensors are allocated on GPU device.
  if(!(A.device().is_cuda() && B.device().is_cuda()))
    throw std::invalid_argument("cutlass_gemm only supports GPU device. Use .to(device=torch.device('cuda'))");

  
  if (A.dtype() != torch::kBFloat16 || B.dtype() != torch::kBFloat16) {
    throw std::invalid_argument("cutlass_gemm only supports bfloat16 type. Use .to(dtype=torch.bfloat16)");
  }

  if (A.stride(1) != 1) {
    throw std::invalid_argument("A must be row major");
  }

  if (B.stride(1) != 1) {
    throw std::invalid_argument("B must be row major");
  }

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
  
  // Create output tensor C with bfloat16 type to match CUTLASS kernel expectations
  torch::Tensor C = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(A.device()));

  // allocate memory fp4 A,B, as well as scales for A,B
  // right now K must be a multiple of 16
  const unsigned int K_dim_sf_a = K / 16;

  cutlass::float_e2m1_t* A_quantized, *B_quantized;
  cutlass::float_ue4m3_t* A_sf, *B_sf;
  CUDA_CHECK_MALLOC(cudaMalloc(&A_quantized, M * K * sizeof(cutlass::float_e2m1_t)));
  CUDA_CHECK_MALLOC(cudaMalloc(&B_quantized, N * K * sizeof(cutlass::float_e2m1_t)));
  
  CUDA_CHECK_MALLOC(cudaMalloc(&A_sf, M * K_dim_sf_a * sizeof(cutlass::float_ue4m3_t)));
  CUDA_CHECK_MALLOC(cudaMalloc(&B_sf, N * K_dim_sf_a * sizeof(cutlass::float_ue4m3_t)));

  cute::Layout layout_A = cute::make_layout(cute::make_shape(M, K, 1), cute::make_stride(K, 1, 0));
  cute::Layout layout_B = cute::make_layout(cute::make_shape(N, K, 1), cute::make_stride(K, 1, 0));
  cute::Layout layout_A_sf = cute::make_layout(cute::make_shape(M, K_dim_sf_a, 1), cute::make_stride(K_dim_sf_a, 1, 0));
  cute::Layout layout_B_sf = cute::make_layout(cute::make_shape(N, K_dim_sf_a, 1), cute::make_stride(K_dim_sf_a, 1, 0));

  auto A_tensor = cute::make_tensor(static_cast<cutlass::bfloat16_t const*>(_A.data_ptr()), layout_A);
  auto B_tensor = cute::make_tensor(static_cast<cutlass::bfloat16_t const*>(_B.data_ptr()), layout_B);
  auto A_quantized_tensor = cute::make_tensor(A_quantized, layout_A);
  auto B_quantized_tensor = cute::make_tensor(B_quantized, layout_B);
  auto A_sf_tensor = cute::make_tensor(A_sf, layout_A_sf);
  auto B_sf_tensor = cute::make_tensor(B_sf, layout_B_sf);

  cutlass_fp4_gemm(M, N, K, 
                   reinterpret_cast<cutlass::float_e2m1_t const*>(A_quantized), 
                   reinterpret_cast<cutlass::float_e2m1_t const*>(B_quantized), 
                   reinterpret_cast<cutlass::float_ue4m3_t const*>(A_sf), 
                   reinterpret_cast<cutlass::float_ue4m3_t const*>(B_sf), 
                   static_cast<cutlass::bfloat16_t*>(C.data_ptr()),
                   bias_ptr);

  cudaFree(A_quantized);
  cudaFree(B_quantized);
  cudaFree(A_sf);
  cudaFree(B_sf);

  return C;
}

// This function is bound to "cutlass_gemm.mm". 
torch::Tensor fp16_gemm(torch::Tensor A,  // A matrix (m x k)
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

  cutlass_fp16_gemm(M, N, K, 
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
  m.def("fp16_gemm", &fp16_gemm, pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("out"), pybind11::arg("bias"));
  m.def("fp4_gemm", &fp4_gemm, pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("bias"));
}
