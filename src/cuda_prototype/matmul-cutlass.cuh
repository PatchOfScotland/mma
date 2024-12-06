/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>


template <
        class ProblemShape, class CtaTiler,
        class TA, class AStride, class ASmemLayout, class TiledCopyAGlobalShared, class TiledCopyASharedRegisters,
        class TB, class BStride, class BSmemLayout, class TiledCopyBGlobalShared, class TiledCopyBSharedRegisters,
        class TC, class CStride, class CSmemLayout, class TiledMma,
        class Alpha, class Beta,
        int num_stages
>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_pipelined(ProblemShape shape_MNK,
            TA const* A, AStride dA,
            TB const* B, BStride dB,
            TC      * C, CStride dC,
            Alpha alpha, Beta beta
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

#if 1
    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                   // (M, N, K)
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});                   // (BLK_M, BLK_N, BLK_K)

    CUTE_STATIC_ASSERT_V(size(copyA_global_shared) == size(tiled_mma));                     // NumThreads
    CUTE_STATIC_ASSERT_V(size(copyB_global_shared) == size(tiled_mma));                     // NumThreads

    static_assert(is_static<ASmemLayout>::value);
    static_assert(is_static<BSmemLayout>::value);
    static_assert(is_static<CSmemLayout>::value);

    CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));  // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler));  // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));  // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));  // BLK_K

    CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), dA));         // dA strides for shape MK
    CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK), dB));         // dB strides for shape NK
    CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));         // dC strides for shape MN
#endif

    Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)

    // Get the appropriate blocks for this thread block
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
    Tensor gA = local_tile(mA, select<0,2>(cta_tiler), select<0,2>(cta_coord));  // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB, select<1,2>(cta_tiler), select<1,2>(cta_coord));  // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, select<0,1>(cta_tiler), select<0,1>(cta_coord));  // (BLK_M,BLK_N)

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

//    TODO: use collective copy?
    ThrCopy thr_copy_a_global_shared = copyA_global_shared.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a_global_shared.partition_S(gA);                            // (CPY,CPY_M,CPY_K,k)
    Tensor tAsA = thr_copy_a_global_shared.partition_D(sA);                            // (CPY,CPY_M,CPY_K)

    ThrCopy thr_copy_b_global_shared = copyB_global_shared.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b_global_shared.partition_S(gB);                            // (CPY,CPY_N,CPY_K,k)
    Tensor tBsB = thr_copy_b_global_shared.partition_D(sB);                            // (CPY,CPY_N,CPY_K)

    ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);
#ifdef SWIZZLE_BACK
    Tensor tCgC = thr_mma.partition_C(sC);                               // (MMA,MMA_M,MMA_N)
#else
    Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)
#endif
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)

#if 1
    CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));                // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));                // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));                // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));                // CPY_K
    CUTE_STATIC_ASSERT_V(  shape(tCrC) ==   shape(tCgC));                // (MMA,MMA_M,MMA_N)
#endif




//    TODO: custom implementation of pipelining?
    // K tiles in global memory
    int k_tile_max = size<3>(tAgA);
    auto k_tile_count = k_tile_max;
    int k_tile = 0;

    // Current pipe index in smem to read from
    int smem_pipe_read  = 0;
    // Current pipe index in smem to write to
    int smem_pipe_write = num_stages - 1;


    // Start async loads for all pipes but the last
    CUTLASS_PRAGMA_UNROLL
    for (int k_pipe = 0; k_pipe < num_stages - 1; ++k_pipe) {
        copy(copyA_global_shared, tAgA(_,_,_,k_tile), tAsA(_,_,_,k_pipe));
        copy(copyB_global_shared, tBgB(_,_,_,k_tile), tBsB(_,_,_,k_pipe));
        cp_async_fence();
        --k_tile_count;
        if (k_tile_count > 0) { ++k_tile; }
    }



    // Create register tensors for the MMA to operate on
    Tensor tCrA  = thr_mma.partition_fragment_A(sA(_,_,0));                    // (MMA,MMA_M,MMA_K)
    Tensor tCrB  = thr_mma.partition_fragment_B(sB(_,_,0));                    // (MMA,MMA_N,MMA_K)

    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(threadIdx.x);
    Tensor tCsA            = smem_thr_copy_A.partition_S(sA);
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);

    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(threadIdx.x);
    Tensor tCsB            = smem_thr_copy_B.partition_S(sB);
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);


    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));             // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));             // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view));            // CPY_K

    CUTE_STATIC_ASSERT_V(size<1>(tCgC) == size<1>(tCrA));                // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(tCgC) == size<1>(tCrB));                // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                // MMA_K

    // Size of the register pipeline
    auto K_BLOCK_MAX = size<2>(tCrA);


    // Clear the accumulators
    clear(tCrC);


    //    TODO: avoid this for clarity?
    Tensor tCsA_p = tCsA(_,_,_,smem_pipe_read);
    Tensor tCsB_p = tCsB(_,_,_,smem_pipe_read);


    // PREFETCH register pipeline
    if (K_BLOCK_MAX > 1) {
        // Wait until our first prefetched tile is loaded in
        cp_async_wait<num_stages - 2>();
        __syncthreads();

        // Prefetch the first rmem from the first k-tile
        copy(smem_tiled_copy_A, tCsA_p(_,_,Int<0>{}), tCrA_copy_view(_,_,Int<0>{}));
        copy(smem_tiled_copy_B, tCsB_p(_,_,Int<0>{}), tCrB_copy_view(_,_,Int<0>{}));
    }


    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > -(num_stages - 1))
    {
        // Pipeline the outer products with a static for loop.
        //
        // Note, the for_each() function is required here to ensure `k_block` is of type Int<x>.
        for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block)
        {
            if (k_block == K_BLOCK_MAX - 1)
            {
                // Slice the smem_pipe_read smem
                tCsA_p = tCsA(_,_,_,smem_pipe_read);
                tCsB_p = tCsB(_,_,_,smem_pipe_read);

                // Commit the smem for smem_pipe_read
                cp_async_wait<num_stages - 2>();
                __syncthreads();
            }

            // Load A, B shmem->regs for k_block+1
            auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;  // static
            copy(smem_tiled_copy_A, tCsA_p(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
            copy(smem_tiled_copy_B, tCsB_p(_,_,k_block_next), tCrB_copy_view(_,_,k_block_next));

            // Copy gmem to smem before computing gemm on each k-pipe
            if (k_block == 0)
            {
                copy(copyA_global_shared, tAgA(_,_,_,k_tile), tAsA(_,_,_,smem_pipe_write));
                copy(copyB_global_shared, tBgB(_,_,_,k_tile), tBsB(_,_,_,smem_pipe_write));
                cp_async_fence();

                // Advance the tile
                --k_tile_count;
                if (k_tile_count > 0) { ++k_tile; }

                // Advance the pipe -- Doing it here accounts for K_BLOCK_MAX = 1 (no rmem pipe)
                smem_pipe_write = smem_pipe_read;
                ++smem_pipe_read;
                smem_pipe_read = (smem_pipe_read == num_stages) ? 0 : smem_pipe_read;
            }

            // Transform before compute
//            TODO: add transforms?
//            cute::transform(tCrA(_,_,k_block), TransformA{});
//            cute::transform(tCrB(_,_,k_block), TransformB{});

            // Thread-level register gemm for k_block
            gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
        });
    }

#ifdef SWIZZLE_BACK
#ifndef SEPARATE_CMEM
    __syncthreads();
#endif
#endif

    // Write back to global with result
    axpby(alpha, tCrC, beta, tCgC);
// TODO: use this?
//    copy(AutoVectorizingCopy{}, tCrC, tCgC);

#ifdef SWIZZLE_BACK
    __syncthreads();
    cooperative_copy<decltype(size(TiledMma{}))::value>(threadIdx.x, sC, gC);
#endif
}
