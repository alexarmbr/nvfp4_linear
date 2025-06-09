#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>

#include <cuda_fp8.h>

using namespace cute;

__device__ __forceinline__ uint8_t quantize_float2_fp4(float2 value) {
    uint32_t result_;
    asm volatile("{ .reg .b8 tmp; cvt.rn.satfinite.e2m1x2.f32 tmp, %1, %2; cvt.u32.u8 %0, tmp; }"
                 : "=r"(result_)
                 : "f"(value.x), "f"(value.y));
    uint8_t result = static_cast<uint8_t>(result_);
    return result;
}


template<class TensorIn, class TensorDataOut, class TensorScaleOut, class ThreadShape, class ValShape>
__global__ void bf16_to_fp4_kernel(TensorIn In,
TensorDataOut DataOutVectorized,
TensorScaleOut ScaleOut,
ThreadShape thread_shape,
ValShape val_shape
){
    constexpr float max_nvfp4 = 6.0f; // 2^(3 - 1) * 1.5 -> this is 1 11 1, exponent bias is 1 and there is no inf representation
    constexpr float max_fp8_e4m3 = 448.0f; // 2^(15 - 7) * 1.75 -> this is 1 1111 110, exponent bias is 7 and this is an inf representation
    constexpr float reciprocal_max_nvfp4 = 1.0f / max_nvfp4;

    // get the tile for this block
    Tensor blockTileInGmem = In(make_coord(_,_), blockIdx.x, blockIdx.y);
    Tensor blockTileOutGmem = DataOutVectorized(make_coord(_,_), blockIdx.x, blockIdx.y);
    Tensor blockTileScaleOutGmem = ScaleOut(make_coord(_,_), blockIdx.x, blockIdx.y);

    // divide the block tile into 2x8 tiles per thread
    Tensor blockTileInGmem_ = zipped_divide(blockTileInGmem, val_shape);
    // Tensor blockTileOutGmem_ = zipped_divide(blockTileOutGmem, val_shape);
    Tensor blockTileOutGmem_ = zipped_divide(blockTileOutGmem, cute::make_shape(cute::Int<2>{}, cute::Int<1>{}));

    // get the tile for this thread
    Tensor threadTileInGmem = blockTileInGmem_(make_coord(_,_), make_coord(threadIdx.y, threadIdx.x));
    Tensor threadTileOutGmem = blockTileOutGmem_(make_coord(_,_), make_coord(threadIdx.y, threadIdx.x));
    Tensor threadTileReg = cute::make_tensor_like(threadTileInGmem);

    using CopyOp = AutoVectorizingCopy;
    using Atom = Copy_Atom<CopyOp, cutlass::bfloat16_t>;
    cute::copy(Atom{}, threadTileInGmem, threadTileReg);

    bool print_debug = threadIdx.x < 2 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0;
    // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    //     printf("DataOutVectorized: \t"); print(DataOutVectorized); printf("\n");
    //     printf("blockTileOutGmem: \t"); print(blockTileOutGmem); printf("\n");
    //     printf("blockTileOutGmem_: \t"); print(blockTileOutGmem_); printf("\n");
    //     printf("threadTileOutGmem: \t"); print(threadTileOutGmem); printf("\n");
    //     printf("threadTileReg: \t"); print(threadTileReg); printf("\n");
    // }

    // each thread computes the max of each of the 2 rows
    cutlass::bfloat16_t row0_max_ = abs(threadTileReg(make_coord(0, 0)));
    cutlass::bfloat16_t row1_max_ = abs(threadTileReg(make_coord(1, 0)));
    #pragma unroll
        for (int i = 1; i < 8; i++){
            row0_max_ = cutlass::fast_max(row0_max_, abs(threadTileReg(make_coord(0, i))));
            row1_max_ = cutlass::fast_max(row1_max_, abs(threadTileReg(make_coord(1, i))));
        }
    
    float row0_max = static_cast<float>(row0_max_);
    float row1_max = static_cast<float>(row1_max_);
    float received_row0_max = 0.0f; float received_row1_max = 0.0f;
    float row0_scale = 0.0f; float row1_scale = 0.0f;

    // each thread recieves row max from 1 thread up
    received_row0_max = __shfl_sync(0xffffffff, row0_max, threadIdx.x + 1);
    received_row1_max = __shfl_sync(0xffffffff, row1_max, threadIdx.x + 1);

    // if (print_debug) {
    //     printf("threadIdx.x: %d, row0_max: %f, row1_max: %f, received_row0_max: %f, received_row1_max: %f\n", threadIdx.x, row0_max, row1_max, received_row0_max, received_row1_max);
    // }

    // each even thread computes a scale factor for a 2 blocks of 16 elements
    // based on the max from its local chunk of 8 elements and the next chunk over
    if (threadIdx.x % 2 == 0) {
        row0_max = fmaxf(row0_max, received_row0_max);
        row1_max = fmaxf(row1_max, received_row1_max);

        // divide by max representable nvfp4 value (6.0)
        row0_scale = row0_max * reciprocal_max_nvfp4;
        row1_scale = row1_max * reciprocal_max_nvfp4;
        
        // output scales are clamped to max_fp8_e4m3 (448.0)
        row0_scale = fminf(row0_scale, max_fp8_e4m3);
        row1_scale = fminf(row1_scale, max_fp8_e4m3);

        // this is slightly faster than the __nv_cvt_float2_to_fp8x2, requires compiling with -DCUDA_PTX_FP8_CVT_ENABLED
        // POTENTIAL ISSUE - this performs cvt.rn.satfinite.e4m3x2 which rounds to nearest even
        // according to this cudnn doc, we want to round to nearest positive here
        // but cvt.rp.satfinite.e4m3x2.f32  does not compile
        // https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/BlockScaling.html
        cutlass::float_ue4m3_t converter;
        cutlass::float_ue4m3_t fp8_scale0 = converter.convert_from_float(row0_scale);
        cutlass::float_ue4m3_t fp8_scale1 = converter.convert_from_float(row1_scale);

        // store the fp8 scales in the output tensor
        blockTileScaleOutGmem(make_coord(threadIdx.y * 2, threadIdx.x / 2)) = fp8_scale0;
        blockTileScaleOutGmem(make_coord((threadIdx.y * 2) + 1, threadIdx.x / 2)) = fp8_scale1;
        
    }

    // each thread recieves scale values from one thread down
    // at this point only the even threads have the correct scale values
    // so only the odd threads need to recieve the scale values from the even threads
    float recieved_row0_scale = __shfl_sync(0xffffffff, row0_scale, threadIdx.x - 1);
    float recieved_row1_scale = __shfl_sync(0xffffffff, row1_scale, threadIdx.x - 1);
    if (threadIdx.x % 2 == 1) {
        row0_scale = recieved_row0_scale;
        row1_scale = recieved_row1_scale;
    }

    // if (print_debug) {
    //     printf("threadIdx.x: %d, row0_scale: %f, row1_scale: %f, recieved_row0_scale: %f, recieved_row1_scale: %f\n", threadIdx.x, row0_scale, row1_scale, recieved_row0_scale, recieved_row1_scale);
    // }

    float row_0_scale_reciprocal = 1.0f / row0_scale;
    float row_1_scale_reciprocal = 1.0f / row1_scale;

    uint32_t packed_row0 = 0;
    uint32_t packed_row1 = 0;
    
    #pragma unroll
        for (int i = 0; i < 8; i++){
            float x = __bfloat162float(threadTileReg(make_coord(0, i)).to_nv_bfloat16());
            float y = __bfloat162float(threadTileReg(make_coord(1, i)).to_nv_bfloat16());

            // divide by scale factor to map values into nvfp4 range
            // the scale factor is max(x_0, x_1, ..., x_16) / MAX_NVFP4
            x *= row_0_scale_reciprocal;
            y *= row_1_scale_reciprocal;

            float2 value = make_float2(x, y);
            
            // this uint8 contains the bits of 2 fp4s packed together
            // like xxxxyyyy
            uint8_t quantized = quantize_float2_fp4(value);

            // seperate them out so that both fp4 values occupy the 4 least significant bits
            // TODO make sure this is correct, are you sure y is in the 4 least significant bits?
            uint8_t x_fp4 = (quantized & 0b11110000) >> 4;
            uint8_t y_fp4 = quantized & 0b00001111;

            // if (print_debug && threadIdx.x == 1) {
            //     cutlass::float_e2m1_t result = cutlass::float_e2m1_t::bitcast(x_fp4);
            //     cutlass::float_e2m1_t result2 = cutlass::float_e2m1_t::bitcast(y_fp4);
            //     printf("x_fp4: %d, y_fp4: %d, result: %f, result2: %f\n", x_fp4, y_fp4, float(result), float(result2));
            // }

            // shift into a uint64
            packed_row0 |= static_cast<uint32_t>(x_fp4) << (i * 4);
            packed_row1 |= static_cast<uint32_t>(y_fp4) << (i * 4);
        }
    
    // each thread writes 8 fp4 values packed into a uint32 back to gmem
    // for two rows
    threadTileOutGmem(make_coord(0,0)) = packed_row0;
    threadTileOutGmem(make_coord(1,0)) = packed_row1;
    
}


void bf16_to_fp4(cutlass::bfloat16_t const* in, cutlass::float_e2m1_t* data_out, cutlass::float_ue4m3_t* scale_out, int in_rows, int in_cols){
    
    // 16 fp4 values share a single scale factor, 
    constexpr int nvfp4_scale_length = 16;
    // in order to vectorize the writing of fp4 values to gmem, we pack 8 of them into a single uint32
    constexpr int output_data_vectorized_length = 8;
    const int scale_cols = in_cols / nvfp4_scale_length;
    const int out_vectorized_cols = in_cols / output_data_vectorized_length;
    
    auto thread_shape = cute::make_shape(cute::Int<4>{}, cute::Int<32>{});
    auto val_shape = cute::make_shape(cute::Int<2>{}, cute::Int<8>{});

    auto block_rows_data = cute::get<0>(thread_shape) * cute::get<0>(val_shape);
    auto block_cols_data = cute::get<1>(thread_shape) * cute::get<1>(val_shape);
    auto block_cols_scale = block_cols_data / cute::Int<nvfp4_scale_length>{};
    auto block_cols_out_vectorized = block_cols_data / cute::Int<output_data_vectorized_length>{};
    
    auto data_block_shape = cute::make_shape(
        block_rows_data,
        block_cols_data
    ); // thread shape * val shape

    auto scale_block_shape = cute::make_shape(
        block_rows_data,
        block_cols_scale
    ); // thread shape * val shape

    auto data_block_shape_vectorized = cute::make_shape(
        block_rows_data,
        block_cols_out_vectorized
    ); // thread shape * val shape

    Tensor In = cute::make_tensor(in, cute::make_layout(cute::make_shape(in_rows, in_cols), LayoutRight{}));
    Tensor DataOutVectorized = cute::make_tensor(reinterpret_cast<uint32_t*>(data_out), cute::make_layout(cute::make_shape(in_rows, out_vectorized_cols), LayoutRight{}));
    Tensor ScaleOut = cute::make_tensor(scale_out, cute::make_layout(cute::make_shape(in_rows, scale_cols), LayoutRight{}));

    Tensor tiledIn = tiled_divide(In, data_block_shape); // ((BM, BN), m/BM, n/BN)
    Tensor tiledDataOutVectorized = tiled_divide(DataOutVectorized, data_block_shape_vectorized); // ((BM, BN), m/BM, n/BN)
    Tensor tiledScaleOut = tiled_divide(ScaleOut, scale_block_shape); // ((BM, BN), m/BM, n/BN)

    int blocks_x = size<1>(tiledIn);
    int blocks_y = size<2>(tiledIn);
    int threads_x = size<1>(thread_shape);
    int threads_y = size<0>(thread_shape);

    dim3 gridDim (blocks_x, blocks_y);
    dim3 blockDim(threads_x, threads_y);
    
    bf16_to_fp4_kernel<<<gridDim, blockDim>>>(tiledIn, tiledDataOutVectorized, tiledScaleOut, thread_shape, val_shape);
}


