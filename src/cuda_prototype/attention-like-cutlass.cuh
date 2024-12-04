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
        class TA, class ALayout, class ASmemLayout, class TiledCopyAGlobalShared, class TiledCopyASharedRegisters,
        class TB, class BLayout, class BSmemLayout, class TiledCopyBGlobalShared, class TiledCopyBSharedRegisters,
        class TC, class CLayout, class CSmemLayout, class TiledMma,
        class Alpha, class Beta
>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
attention_like_simple(TA const* As, ALayout layoutAs,
            TB const* Bss, BLayout layoutBss,
            TC      * Cs, CLayout layoutCs,
            Alpha alpha, Beta beta
)
{
    using namespace cute;

    extern __shared__ __align__(128) uint1_t shared[];

    ASmemLayout sA_layout;
    BSmemLayout sB_layout;

    TiledCopyAGlobalShared copyA_global_shared;
    TiledCopyBGlobalShared copyB_global_shared;
    TiledCopyASharedRegisters smem_tiled_copy_A;
    TiledCopyBSharedRegisters smem_tiled_copy_B;
    TiledMma tiled_mma;

#if 1
    CUTE_STATIC_ASSERT_V(size(copyA_global_shared) == size(tiled_mma));                     // NumThreads
    CUTE_STATIC_ASSERT_V(size(copyB_global_shared) == size(tiled_mma));                     // NumThreads

    static_assert(is_static<ASmemLayout>::value);
    static_assert(is_static<BSmemLayout>::value);
    static_assert(is_static<CSmemLayout>::value);
#endif

    Tensor mAs = make_tensor(make_gmem_ptr(As), layoutAs);
    Tensor mBss = make_tensor(make_gmem_ptr(Bss), layoutBss);
    Tensor mCs = make_tensor(make_gmem_ptr(Cs), layoutCs);

    Tensor gA = mAs(_, _, blockIdx.x);
    Tensor gBs = mBss(_, _, blockIdx.x, _);
    Tensor gC = mCs(_, _, blockIdx.x);

    // Shared memory buffers
    auto smemA = reinterpret_cast<TA *>(shared);
    auto smemB = reinterpret_cast<TB *>(smemA + cosize_v<ASmemLayout>);
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);            // (BLK_M,BLK_K)
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);            // (BLK_N,BLK_K)

    ThrCopy thr_copy_a_global_shared = copyA_global_shared.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a_global_shared.partition_S(gA);
    Tensor tAsA = thr_copy_a_global_shared.partition_D(sA);

    ThrCopy thr_copy_b_global_shared = copyB_global_shared.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b_global_shared.partition_S(gBs);
    Tensor tBsB = thr_copy_b_global_shared.partition_D(sB);

    ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);
    Tensor tCgC = thr_mma.partition_C(gC);
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);

#if 1
    CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));                // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));                // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));                // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));                // CPY_K
    CUTE_STATIC_ASSERT_V(  shape(tCrC) ==   shape(tCgC));                // (MMA,MMA_M,MMA_N)
#endif

    // Create register tensors for the MMA to operate on
    Tensor tCrA  = thr_mma.partition_fragment_A(sA);                    // (MMA,MMA_M,MMA_K)
    Tensor tCrB  = thr_mma.partition_fragment_B(sB);                    // (MMA,MMA_N,MMA_K)

    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(threadIdx.x);
    Tensor tCsA            = smem_thr_copy_A.partition_S(sA);
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);

    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(threadIdx.x);
    Tensor tCsB            = smem_thr_copy_B.partition_S(sB);
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);


//    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));             // CPY_M
//    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));             // CPY_K
//    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // CPY_N
//    CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCrB_copy_view));            // CPY_K
//
//    CUTE_STATIC_ASSERT_V(size<1>(tCgC) == size<1>(tCrA));                // MMA_M
//    CUTE_STATIC_ASSERT_V(size<2>(tCgC) == size<1>(tCrB));                // MMA_N
//    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                // MMA_K

    // Clear the accumulators
    clear(tCrC);

//    if (thread0() && block0()) {
//        print(gA);
//        print("\n");
//        print(gBs);
//        print("\n");
//        print(gC);
//        print("\n");
//        print(tAgA);
//        print("\n");
//        print(tBgB);
//        print("\n");
//    }

    copy(copyA_global_shared, tAgA, tAsA);

    int k_tile_max = size<3>(tBgB);
    for (int k_tile = 0; k_tile < k_tile_max; k_tile++)
    {
        // Copy global -> shared
        __syncthreads();
        copy(copyB_global_shared, tBgB(_,_,_,k_tile), tBsB);
#ifndef SYNC_CPY
        cp_async_fence();
        cp_async_wait<0>();
#endif
        __syncthreads();

        // Inner loop
        constexpr int K_BLOCK_MAX = size<2>(tCrA);
        CUTE_UNROLL
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
        {
            // Copy shared -> registers
            copy(smem_tiled_copy_A, tCsA(_,_,k_block), tCrA_copy_view(_,_,k_block));
            copy(smem_tiled_copy_B, tCsB(_,_,k_block), tCrB_copy_view(_,_,k_block));

            // GEMM on k_block in registers
            gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
        }
    }

    // Write back to global with result
    axpby(alpha, tCrC, beta, tCgC);
}
