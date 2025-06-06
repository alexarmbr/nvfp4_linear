#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>

#include <cuda_fp8.h>

using namespace cute;


template<class TensorIn, class TensorDataOut, class TensorScaleOut, class ThreadShape, class ValShape>
__global__ void bf16_to_fp4_kernel(TensorIn In,
TensorDataOut DataOut,
TensorScaleOut ScaleOut,
ThreadShape thread_shape,
ValShape val_shape
){
    constexpr float max_nvfp4 = 6.0f; // 2^(3 - 1) * 1.5 -> this is 1 11 1, exponent bias is 1 and there is no inf representation
    constexpr float max_fp8_e4m3 = 448.0f; // 2^(15 - 7) * 1.75 -> this is 1 1111 110, exponent bias is 7 and this is an inf representation
    constexpr float reciprocal_max_nvfp4 = 1.0f / max_nvfp4;
    constexpr float reciprocal_max_fp8_e4m3 = 1.0f / max_fp8_e4m3;

    // get the tile for this block
    Tensor blockTileInGmem = In(make_coord(_,_), blockIdx.x, blockIdx.y);
    Tensor blockTileOutGmem = DataOut(make_coord(_,_), blockIdx.x, blockIdx.y);
    Tensor blockTileScaleOutGmem = ScaleOut(make_coord(_,_), blockIdx.x, blockIdx.y);

    // divide the block tile into 2x8 tiles per thread
    Tensor blockTileInGmem_ = zipped_divide(blockTileInGmem, val_shape);
    Tensor blockTileOutGmem_ = zipped_divide(blockTileOutGmem, val_shape);

    // get the tile for this thread
    Tensor threadTileInGmem = blockTileInGmem_(make_coord(_,_), make_coord(threadIdx.y, threadIdx.x));
    Tensor threadTileOutGmem = blockTileOutGmem_(make_coord(_,_), make_coord(threadIdx.y, threadIdx.x));
    Tensor threadTileReg = cute::make_tensor_like(threadTileInGmem);

    using CopyOp = AutoVectorizingCopy;
    using Atom = Copy_Atom<CopyOp, cutlass::bfloat16_t>;
    cute::copy(Atom{}, threadTileInGmem, threadTileReg);

    // each thread computes the max of each of the 2 rows
    cutlass::bfloat16_t row0_max = threadTileReg(make_coord(0, 0));
    cutlass::bfloat16_t row1_max = threadTileReg(make_coord(1, 0));
    #pragma unroll
        for (int i = 1; i < 8; i++){
            row0_max = cutlass::fast_max(row0_max, threadTileReg(make_coord(0, i)));
            row1_max = cutlass::fast_max(row1_max, threadTileReg(make_coord(1, i)));
        }

    // Warp shuffle: odd threads send their values to even threads (threadIdx.x - 1)
    // Even threads receive and take max with their own values
    float row0_max_float = static_cast<float>(row0_max);
    float row1_max_float = static_cast<float>(row1_max);
    
    // Fix warp shuffle to handle boundary case properly
    // Even threads receive from odd threads, but handle threadIdx.x = 30 case
    float received_row0_max, received_row1_max;
    if (threadIdx.x % 2 == 0 && threadIdx.x + 1 < 32) {
        received_row0_max = __shfl_sync(0xffffffff, row0_max_float, threadIdx.x + 1);
        received_row1_max = __shfl_sync(0xffffffff, row1_max_float, threadIdx.x + 1);
    } else {
        received_row0_max = 0.0f;  // or row0_max_float if you want to use own value
        received_row1_max = 0.0f;  // or row1_max_float if you want to use own value
    }
    
    // Even threads take max between received value and their own
    
    // TODO look at what shfl_sync does
    if (threadIdx.x % 2 == 0) {
        if (threadIdx.x + 1 < 32) {
            row0_max = cutlass::fast_max(row0_max, static_cast<cutlass::bfloat16_t>(received_row0_max));
            row1_max = cutlass::fast_max(row1_max, static_cast<cutlass::bfloat16_t>(received_row1_max));
        }
        // If threadIdx.x = 30, just use own values (no adjacent odd thread)

        // divide by max representable nvfp4 value (6.0)
        float row0_scale = row0_max * reciprocal_max_nvfp4;
        float row1_scale = row1_max * reciprocal_max_nvfp4;
        
        // output scales are clamped to max_fp8_e4m3
        row0_scale = fminf(row0_scale, max_fp8_e4m3);
        row1_scale = fminf(row1_scale, max_fp8_e4m3);
        float2 row_scales = make_float2(row0_scale, row1_scale);

        // this is slightly faster than the __nv_cvt_float2_to_fp8x2, requires compiling with -DCUDA_PTX_FP8_CVT_ENABLED
        cutlass::float_ue4m3_t converter;
        cutlass::float_ue4m3_t fp8_scale0 = converter.convert_from_float(row0_scale);
        cutlass::float_ue4m3_t fp8_scale1 = converter.convert_from_float(row1_scale);
        
        
        // __nv_fp8x2_storage_t fp8_scales = __nv_cvt_float2_to_fp8x2(row_scales, 
        //                                                            __NV_SATFINITE, 
        //                                                            __NV_E4M3);
        // unsigned char fp8_scale0_raw = (unsigned char)(fp8_scales & 0xFF);        // lower 8 bits
        // unsigned char fp8_scale1_raw = (unsigned char)((fp8_scales >> 8) & 0xFF); // upper 8 bits
        // cutlass::float_ue4m3_t fp8_scale0;
        // cutlass::float_ue4m3_t fp8_scale1;
        // fp8_scale0.storage = fp8_scale0_raw;
        // fp8_scale1.storage = fp8_scale1_raw;

        // store the fp8 scales in the output tensor
        blockTileScaleOutGmem(make_coord(threadIdx.y * 2, threadIdx.x / 2)) = fp8_scale0;
        blockTileScaleOutGmem(make_coord(threadIdx.y * 2 + 1, threadIdx.x / 2)) = fp8_scale1;
        
        // Debug: print which thread is storing scales
        if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x < 4) {
            printf("Thread (%d, %d) storing scales at rows (%d, %d), col %d\n", 
                   threadIdx.y, threadIdx.x, 
                   threadIdx.y * 2, threadIdx.y * 2 + 1, threadIdx.x / 2);
        }
    }

    if (thread0() && blockIdx.x == 0 && blockIdx.y == 0)
    {
        printf("ScaleOut: \t"); print(ScaleOut.layout()); printf("\n");
        printf("Block Tile Scale Out: \t"); print(blockTileScaleOutGmem.layout()); printf("\n");
    }
        // TODO: Store fp8 scales appropriately
        // For now, we'll skip storing scales and focus on data conversion
        
        // Scale the data
        // for (int i = 0; i < 8; i++){
        // scale the data
        // for (int i = 0; i < 8; i++){
        //     threadTileReg(make_coord(0, i)) = threadTileReg(make_coord(0, i)) * row0_scale;
        //     threadTileReg(make_coord(1, i)) = threadTileReg(make_coord(1, i)) * row1_scale;
        // }
        // }

    // cute::copy(Atom{}, threadTileReg, threadTileOutGmem);

    // cutlass::float_ue4m3_t fp8_scale0_converter;
    // cutlass::float_ue4m3_t fp8_scale1_converter;
    // cutlass::float_ue4m3_t fp8_scale0 = fp8_scale0_converter.convert_from_float(row0_scale);
    // cutlass::float_ue4m3_t fp8_scale1 = fp8_scale1_converter.convert_from_float(row1_scale);

    // // store the fp8 scales in the output tensor
    // blockTileScaleOutGmem(make_coord(threadIdx.y, threadIdx.x / 2)) = fp8_scale0;
    // blockTileScaleOutGmem(make_coord(threadIdx.y + 1, threadIdx.x / 2)) = fp8_scale1;



    // cutlass::bfloat16_t threadTile[2][8]

    // Tensor this_thr_tile_In = thr_tile_In(make_coord(_,_), threadIdx.x);
    // Tensor this_thr_tile_DataOut = thr_tile_DataOut(make_coord(_,_), threadIdx.x);

    // if (thread0())
    // {
    //     // print all layouts
    //     printf("In Layout: \t"); print(In.layout()); printf("\n");
    //     printf("Block Tile In Layout: \t"); print(blockTileInGmem.layout()); printf("\n");
    //     printf("Divided Block Tile In Layout: \t"); print(blockTileInGmem_.layout()); printf("\n");
    //     printf("Thread Tile In Layout: \t"); print(threadTileInGmem.layout()); printf("\n");
    //     printf("Thread Tile Out Layout: \t"); print(threadTileOutGmem.layout()); printf("\n");
    //     printf("Thread Tile Reg Layout: \t"); print(threadTileReg.layout()); printf("\n");
    // }
    
}


// void bf16_to_fp4(cutlass::bfloat16_t const* in, cutlass::float_e2m1_t* data_out, cutlass::float_ue4m3_t* scale_out, int rows, int cols){
void bf16_to_fp4(cutlass::bfloat16_t const* in, cutlass::bfloat16_t* data_out, cutlass::float_ue4m3_t* scale_out, int rows, int cols){

    constexpr int nvfp4_scale_vector_length = 16;
    // for now, assert that cols is divisible by nvfp4_scale_vector_length
    assert(cols % nvfp4_scale_vector_length == 0);
    
    auto thread_shape = cute::make_shape(cute::Int<4>{}, cute::Int<32>{});
    auto val_shape = cute::make_shape(cute::Int<2>{}, cute::Int<8>{});

    auto block_rows_data = cute::get<0>(thread_shape) * cute::get<0>(val_shape);
    auto block_cols_data = cute::get<1>(thread_shape) * cute::get<1>(val_shape);
    auto block_cols_scale = block_cols_data / cute::Int<nvfp4_scale_vector_length>{};
    
    auto data_block_shape = cute::make_shape(
        block_rows_data,
        block_cols_data
    ); // thread shape * val shape

    auto scale_block_shape = cute::make_shape(
        block_rows_data,
        block_cols_scale
    ); // thread shape * val shape

    Tensor In = cute::make_tensor(in, cute::make_layout(cute::make_shape(rows, cols), LayoutRight{}));
    Tensor DataOut = cute::make_tensor(data_out, cute::make_layout(cute::make_shape(rows, cols), LayoutRight{}));
    Tensor ScaleOut = cute::make_tensor(scale_out, cute::make_layout(cute::make_shape(rows, cols / nvfp4_scale_vector_length), LayoutRight{}));
 
    Tensor tiledIn = tiled_divide(In, data_block_shape); // ((BM, BN), m/BM, n/BN)
    Tensor tiledDataOut = tiled_divide(DataOut, data_block_shape); // ((BM, BN), m/BM, n/BN)
    Tensor tiledScaleOut = tiled_divide(ScaleOut, scale_block_shape); // ((BM, BN), m/BM, n/BN)
    
    
    // Define CopyOp using AutoVectorizingCopy for flexibility
    // using CopyOp = AutoVectorizingCopy;
    // // using CopyOp2 = UniversalCopy<uint_byte_t<16>>;
    // using Atom = Copy_Atom<CopyOp, cutlass::bfloat16_t>;
    // auto tiledCopy = make_tiled_copy(Atom{}, thread_layout, val_layout);

    int blocks_x = size<1>(tiledIn);
    int blocks_y = size<2>(tiledIn);
    int threads_x = size<1>(thread_shape);
    int threads_y = size<0>(thread_shape);

    // printf("tiled in layout: \t"); print(tiledIn.layout()); printf("\n");

    // printf("Blocks X: %d, Blocks Y: %d\n", blocks_x, blocks_y);
    // printf("Threads X: %d, Threads Y: %d\n", threads_x, threads_y);

    dim3 gridDim (blocks_x, blocks_y);
    dim3 blockDim(threads_x, threads_y);
    
    bf16_to_fp4_kernel<<<gridDim, blockDim>>>(tiledIn, tiledDataOut, tiledScaleOut, thread_shape, val_shape);
}


