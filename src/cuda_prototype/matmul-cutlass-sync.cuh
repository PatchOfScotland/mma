#include <cute/tensor.hpp>


template <
        class ProblemShape, class CtaTiler,
        class TA, class AStride, class ASmemLayout, class TiledCopyAGlobalShared, class TiledCopyASharedRegisters,
        class TB, class BStride, class BSmemLayout, class TiledCopyBGlobalShared, class TiledCopyBSharedRegisters,
        class TC, class CStride, class CSmemLayout, class TiledMma
>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_sync_cpy(ProblemShape shape_MNK,
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
    TiledCopyASharedRegisters smem_tiled_copy_A;
    TiledCopyBSharedRegisters smem_tiled_copy_B;
    TiledMma tiled_mma;

    Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA);
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB);
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC);

    // Get the appropriate blocks for this thread block
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, select<0,2>(cta_tiler), select<0,2>(cta_coord));
    Tensor gB = local_tile(mB, select<1,2>(cta_tiler), select<1,2>(cta_coord));
    Tensor gC = local_tile(mC, select<0,1>(cta_tiler), select<0,1>(cta_coord));

    // Shared memory buffers
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
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);

    // Create register tensors for the MMA to operate on
    Tensor tCrA  = thr_mma.partition_fragment_A(sA);
    Tensor tCrB  = thr_mma.partition_fragment_B(sB);

    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(threadIdx.x);
    Tensor tCsA            = smem_thr_copy_A.partition_S(sA);
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);

    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(threadIdx.x);
    Tensor tCsB            = smem_thr_copy_B.partition_S(sB);
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);

    // Clear the accumulators
    clear(tCrC);

    // smem copy registers
    Tensor tArA = make_fragment_like(tAsA);
    Tensor tBrB = make_fragment_like(tBsB);

    // Copy gmem to rmem for k_tile=0
    copy(copyA_global_shared, tAgA(_,_,_,0), tArA);
    copy(copyB_global_shared, tBgB(_,_,_,0), tBrB);

    // Copy rmem to smem
    copy(tArA, tAsA);
    copy(tBrB, tBsB);
    __syncthreads();

    // Load A, B shmem->regs for k_block=0
    copy(smem_tiled_copy_A, tCsA(_,_,0), tCrA_copy_view(_,_,0));
    copy(smem_tiled_copy_B, tCsB(_,_,0), tCrB_copy_view(_,_,0));
    auto K_TILE_MAX  = size<3>(tAgA);
    auto K_BLOCK_MAX = size<2>(tCrA);


    CUTE_NO_UNROLL
    for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
    {
        // Pipeline the k-mode of the block registers
        CUTE_UNROLL
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
        {
            if (k_block == K_BLOCK_MAX - 1)
            {
                // Copy rmem to smem
                __syncthreads();
                copy(tArA, tAsA);
                copy(tBrB, tBsB);
                __syncthreads();
            }

            // Copy smem to rmem for k_block+1
            int k_block_next = (k_block + 1) % K_BLOCK_MAX;
            copy(smem_tiled_copy_A, tCsA(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
            copy(smem_tiled_copy_B, tCsB(_,_,k_block_next), tCrB_copy_view(_,_,k_block_next));
            if (k_block == 0)
            {
                // Copy gmem to rmem for k_tile+1
                int k_tile_next = (k_tile + 1 < K_TILE_MAX) ? k_tile + 1 : k_tile;
                copy(copyA_global_shared, tAgA(_,_,_,k_tile_next), tArA);
                copy(copyB_global_shared, tBgB(_,_,_,k_tile_next), tBrB);
            }
            // Thread-level register gemm for k_block
            gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
        }
    }

#ifdef SWIZZLE_BACK
#ifndef SEPARATE_CMEM
    cp_async_wait<0>();
    __syncthreads();
#endif
#endif

    copy(AutoVectorizingCopy{}, tCrC, tCgC);

#ifdef SWIZZLE_BACK
    __syncthreads();
    cooperative_copy<decltype(size(TiledMma{}))::value, 128>(threadIdx.x, sC, gC);
#endif
}
