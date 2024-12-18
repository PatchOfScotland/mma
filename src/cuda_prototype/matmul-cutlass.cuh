// Written with inspiration and snippets from https://github.com/NVIDIA/cutlass/tree/main/examples/cute/tutorial
#include <cute/tensor.hpp>


template <
    class ProblemShape, class CtaTiler,
    class TA, class AStride, class ASmemLayout, class TiledCopyAGlobalShared, class TiledCopyASharedRegisters,
    class TB, class BStride, class BSmemLayout, class TiledCopyBGlobalShared, class TiledCopyBSharedRegisters,
    class TC, class CStride, class CSmemLayout, class TiledMma,
    int num_stages
>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_pipelined(
    ProblemShape shape_MNK,
    TA const* A, AStride dA,
    TB const* B, BStride dB,
    TC      * C, CStride dC
)
{
    using namespace cute;

    extern __shared__ __align__(128) uint1_t shared[];

    ASmemLayout sA_layout;
    BSmemLayout sB_layout;

    CtaTiler cta_tiler;

    TiledCopyAGlobalShared copyA_global_shared;
    TiledCopyBGlobalShared copyB_global_shared;
    TiledCopyASharedRegisters copyA_shared_registers;
    TiledCopyBSharedRegisters copyB_shared_registers;
    TiledMma tiled_mma;

    // Global memory tensors
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA);
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB);
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC);

    // Block tiles
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, select<0,2>(cta_tiler), select<0,2>(cta_coord));
    Tensor gB = local_tile(mB, select<1,2>(cta_tiler), select<1,2>(cta_coord));
    Tensor gC = local_tile(mC, select<0,1>(cta_tiler), select<0,1>(cta_coord));

    // Shared memory tensors
    auto smemA = reinterpret_cast<TA *>(shared);
    auto smemB = reinterpret_cast<TB *>(smemA + cosize_v<ASmemLayout>);
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);

#ifdef SWIZZLE_BACK
    CSmemLayout sC_layout;

#ifdef SEPARATE_CMEM
    auto smemC = reinterpret_cast<TC *>(smemB + cosize_v<BSmemLayout>);
#else
    auto smemC = reinterpret_cast<TC *>(smemA);
#endif

    Tensor sC = make_tensor(make_smem_ptr(smemC), sC_layout);
#endif

    // Thread tiles for global->shared copy
    ThrCopy thr_copy_a_global_shared = copyA_global_shared.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a_global_shared.partition_S(gA);
    Tensor tAsA = thr_copy_a_global_shared.partition_D(sA);

    ThrCopy thr_copy_b_global_shared = copyB_global_shared.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b_global_shared.partition_S(gB);
    Tensor tBsB = thr_copy_b_global_shared.partition_D(sB);

    ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);

#ifdef SWIZZLE_BACK
    Tensor tCgC = thr_mma.partition_C(sC);
#else
    Tensor tCgC = thr_mma.partition_C(gC);
#endif

    int k_tile_max = size<3>(tAgA);
    auto k_tiles_left = k_tile_max;
    int k_tile = 0;

    int read_stage  = 0;
    int write_stage = num_stages - 1;
    
    // Start async copies for all stages except the write_stage
    CUTLASS_PRAGMA_UNROLL
    for (int stage = 0; stage < num_stages - 1; ++stage) {
        copy(copyA_global_shared, tAgA(_,_,_,k_tile), tAsA(_,_,_,stage));
        copy(copyB_global_shared, tBgB(_,_,_,k_tile), tBsB(_,_,_,stage));
        cp_async_fence();

        --k_tiles_left;
        if (k_tiles_left > 0) {
            ++k_tile;
        }
    }

    // Register tensors
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);

    Tensor tCrA  = thr_mma.partition_fragment_A(sA(_,_,0));
    Tensor tCrB  = thr_mma.partition_fragment_B(sB(_,_,0));

    // Thread tiles for shared->registers copy
    auto smem_thr_copy_A   = copyA_shared_registers.get_thread_slice(threadIdx.x);
    Tensor tCsA            = smem_thr_copy_A.partition_S(sA);
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);

    auto smem_thr_copy_B   = copyB_shared_registers.get_thread_slice(threadIdx.x);
    Tensor tCsB            = smem_thr_copy_B.partition_S(sB);
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);
    
    // Clear result registers
    clear(tCrC);

    Tensor tCsA_read = tCsA(_,_,_,read_stage);
    Tensor tCsB_read = tCsB(_,_,_,read_stage);

    constexpr int K_BLOCK_MAX = size<2>(tCrA);
    if (K_BLOCK_MAX > 1) {
        // Wait for first async copy to complete
        cp_async_wait<num_stages - 2>();
        __syncthreads();

        // Copy first k_block shared->registers
        copy(copyA_shared_registers, tCsA_read(_,_,Int<0>{}), tCrA_copy_view(_,_,Int<0>{}));
        copy(copyB_shared_registers, tCsB_read(_,_,Int<0>{}), tCrB_copy_view(_,_,Int<0>{}));
    }

    CUTE_NO_UNROLL
    while (k_tiles_left > -(num_stages - 1))
    {
        CUTE_UNROLL
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
        {
            if (k_block == K_BLOCK_MAX - 1)
            {
                // Update read_stage tensors
                tCsA_read = tCsA(_,_,_,read_stage);
                tCsB_read = tCsB(_,_,_,read_stage);

                // Wait for next async copy to complete
                cp_async_wait<num_stages - 2>();
                __syncthreads();
            }

            // Copy shared->registers for next k_block
            auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;
            copy(copyA_shared_registers, tCsA_read(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
            copy(copyB_shared_registers, tCsB_read(_,_,k_block_next), tCrB_copy_view(_,_,k_block_next));

            if (k_block == 0)
            {
                // Copy global->shared for next k_tile
                copy(copyA_global_shared, tAgA(_,_,_,k_tile), tAsA(_,_,_,write_stage));
                copy(copyB_global_shared, tBgB(_,_,_,k_tile), tBsB(_,_,_,write_stage));
                cp_async_fence();

                --k_tiles_left;
                if (k_tiles_left > 0) {
                    ++k_tile;
                }

                // Update read and write stages
                write_stage = read_stage;
                ++read_stage;
                read_stage = (read_stage == num_stages) ? 0 : read_stage;
            }

            // Perform mma on k_block in registers
            gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
        }
    }

#ifdef SWIZZLE_BACK
#ifndef SEPARATE_CMEM
    cp_async_wait<0>();
    __syncthreads();
#endif
#endif

    // Copy result registers->global
    copy(AutoVectorizingCopy{}, tCrC, tCgC);

#ifdef SWIZZLE_BACK
    __syncthreads();
    cooperative_copy<decltype(size(TiledMma{}))::value, 128>(threadIdx.x, sC, gC);
#endif
}
