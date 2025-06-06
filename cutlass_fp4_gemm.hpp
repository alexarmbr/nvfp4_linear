/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/


// CUTLASS 3.X syntax GEMM
// Adapted from https://github.com/NVIDIA/cutlass/blob/main/examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm.cu

#include <stdexcept>
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

using nvfp4_t = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using DataType = nvfp4_t::DataType;
using ScaleFactorType = nvfp4_t::ScaleFactorType;

void cutlass_fp4_gemm(int M, int N, int K, DataType const* ptrA, DataType const* ptrB, ScaleFactorType const* ptrA_sf, ScaleFactorType const* ptrB_sf, cutlass::bfloat16_t* ptrC, float const* ptrBias) {

    // A matrix configuration
    using         ElementA    = nvfp4_t;    // Element type for A matrix operand
    using         LayoutATag  = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
    constexpr int AlignmentA  = 32;                                             // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

    // B matrix configuration
    using         ElementB    = nvfp4_t;    // Element type for B matrix operand
    using         LayoutBTag  = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
    constexpr int AlignmentB  = 32;                                             // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

    // C/D matrix configuration
    using         ElementD    = cutlass::bfloat16_t;                            // Element type for D matrix operand
    using         ElementC    = cutlass::bfloat16_t;                            // Element type for C matrix operand
    using         LayoutCTag  = cutlass::layout::RowMajor;                      // Layout type for C matrix operand
    using         LayoutDTag  = cutlass::layout::RowMajor;                      // Layout type for D matrix operand
    constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
    constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
    
    // Kernel functional config
    using ElementAccumulator  = float;                                          // Element type for internal accumulation
    using ArchTag             = cutlass::arch::Sm100;                           // Tag indicating the minimum SM that supports the intended feature
    using OperatorClass       = cutlass::arch::OpClassBlockScaledTensorOp;      // Operator class tag

    // Kernel Perf config
    using MmaTileShape        = Shape<_256,_256,_256>;                          // MMA's tile size
    using ClusterShape        = Shape<_4,_4,_1>;                                // Shape of the threadblocks in a cluster

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,                      
        MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutCTag, AlignmentC,
        ElementD, LayoutDTag, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto                      // Epilogue schedule policy
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementA, LayoutATag, AlignmentA,
        ElementB, LayoutBTag, AlignmentB,
        ElementAccumulator,
        MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto                             // Kernel schedule policy. Auto or using targeted scheduling policy
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int,int>,                                                   // Indicates ProblemShape
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  Gemm gemm_op;
  cutlass::Status status;

  //
  // Define the problem size
  //

  float alpha = 1.00f;
  float beta = 0.0f;

    using StrideA   = typename Gemm::GemmKernel::StrideA;
    using LayoutA   = decltype(cute::make_layout(make_shape(0,0,0), StrideA{}));
    using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;      // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.
    using StrideB   = typename Gemm::GemmKernel::StrideB;
    using LayoutB   = decltype(cute::make_layout(make_shape(0,0,0), StrideB{}));
    using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;      // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.
    using StrideC   = typename Gemm::GemmKernel::StrideC;
    using LayoutC   = decltype(cute::make_layout(make_shape(0,0,0), StrideC{}));
    using StrideD   = typename Gemm::GemmKernel::StrideD;
    using LayoutD   = decltype(cute::make_layout(make_shape(0,0,0), StrideD{}));


    /// Initialization
    StrideA stride_A;
    LayoutA layout_A;
    LayoutSFA layout_SFA;
    StrideB stride_B;
    LayoutB layout_B;
    LayoutSFB layout_SFB;
    StrideC stride_C;
    LayoutC layout_C;
    uint64_t seed;
  
  stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});

using Sm1xxBlkScaledConfig =  typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
  layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));
  //
  // Launch GEMM on the device
  //
  
  typename Gemm::Arguments arguments {
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, 1},
    { // Mainloop arguments
      ptrA, stride_A,
      ptrB, stride_B,
      ptrA_sf, layout_SFA,
      ptrB_sf, layout_SFB
    },
    { // Epilogue arguments
      {alpha, beta},
      ptrC, stride_C,
      ptrC, stride_C
    }
  };
  
  // Note: Bias support may need to be implemented through a custom epilogue fusion
  // For now, we'll skip bias support in the CUTLASS kernel
  
  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check if the problem size is supported or not
  status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("Problem size not supported by CUTLASS kernel");
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("Failed to initialize CUTLASS kernel");
  }

  // Correctness / Warmup iteration
  status = gemm_op.run();
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS kernel execution failed");
  }

}





