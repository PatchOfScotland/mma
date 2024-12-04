#include <cstdio>
#include <mma.h>
#include "matmul.cuh"
#include "helpers.cuh"
#include "matmul-tensor-naive.cuh"
#include "matmul-tensor.cuh"
#include "matmul-cutlass.cuh"
#include "matmul-cutlass-simple.cuh"
#include "attention-like-cutlass.cuh"
#include "matmul-cutlass-sync.cuh"
#include "matmul-cutlass2.cuh"
//#include "cuda_fp16.h"
#include "cutlass/half.h"
#include <cassert>
//#include <cublas.h>
#include <cublas_v2.h>
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/kernel/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_epilogue.hpp"

//#include "../../cutlass/test/unit/gemm/device/default_gemm_configuration.hpp"
//#include "../../cutlass/test/unit/cute/cooperative_gemm_common.hpp"


#define WARP_SIZE 32
#define SHARED_MEM_SIZE 49152
#define MAX_THREADS_PER_BLOCK 1024
#define MAX_REGISTERS_PER_BLOCK 65536

#ifndef SHARED_PADDING
#define SHARED_PADDING 8
#endif

#ifndef NUM_STAGES
#define NUM_STAGES 2
#endif

// Set constants using compiler options
#ifndef WMMA_M
#define WMMA_M 16
#endif
#ifndef WMMA_N
#define WMMA_N 16
#endif
#ifndef WMMA_K
#define WMMA_K 16
#endif
#ifndef FRAGS_M
#define FRAGS_M 4
#endif
#ifndef FRAGS_N
#define FRAGS_N 4
#endif
#ifndef FRAGS_K
#define FRAGS_K 1
#endif
#ifndef WARP_TILES_M
#define WARP_TILES_M 1
#endif
#ifndef WARP_TILES_N
#define WARP_TILES_N 1
#endif
#ifndef WARP_TILES_K
#define WARP_TILES_K 4
#endif
#ifndef BLOCK_TILES_M
#define BLOCK_TILES_M 2
#endif
#ifndef BLOCK_TILES_N
#define BLOCK_TILES_N 2
#endif

typedef cutlass::half_t half_t;

#ifdef ELM_T
typedef ELM_T element_type;
#else
typedef half_t element_type;
#endif

#ifdef ACC_T
typedef ACC_T acc_type;
#else
typedef float acc_type;
#endif


enum mm_kernel {
    register_tiled,
    tensor_naive,
    tensor_optimized,
    cublas,
    cutlass_default,
    cutlass_custom,
    cute_mm,
    cutlass_simple
};

enum cutlass_version
{
    DEFAULT,
    PADDING,
    VECTORIZED
};


template <typename elmT, typename elmAccT = elmT>
long int benchmark_optimized_tensor_mmm(
    int n_runs,
    elmT *A_device,
    elmT *B_device,
    elmAccT *C_device,
    int m,
    int n,
    int k)
{
    constexpr unsigned int threads_per_block = BLOCK_TILES_M * BLOCK_TILES_N * WARP_SIZE;
    printf("    Threads used: %d/%d\n", threads_per_block, MAX_THREADS_PER_BLOCK);
    assert(threads_per_block <= MAX_THREADS_PER_BLOCK);
    // Assumes num_warps >= block_tiles_m * block_tiles_n, i.e. all block tiles are handled by a warp
    assert(threads_per_block / WARP_SIZE >= BLOCK_TILES_M * BLOCK_TILES_N);

    printf("    Using wmma %d x %d x %d\n", WMMA_M, WMMA_N, WMMA_K);
    printf("    Using frags %d x %d x %d\n", FRAGS_M, FRAGS_N, FRAGS_K);
    printf("    Using warp tiles %d x %d x %d\n", WARP_TILES_M, WARP_TILES_N, WARP_TILES_K);
    printf("    Using block tiles %d x %d\n", BLOCK_TILES_M, BLOCK_TILES_N);

    constexpr unsigned int shared_m = WMMA_M * FRAGS_M * WARP_TILES_M * BLOCK_TILES_M;
    constexpr unsigned int shared_n = WMMA_N * FRAGS_N * WARP_TILES_N * BLOCK_TILES_N;
    constexpr unsigned int shared_k = WMMA_K * FRAGS_K * WARP_TILES_K;

    int dimx = ceil(((float) n)/(shared_n));
    int dimy = ceil(((float) m)/(shared_m));

    dim3 grid(dimx, dimy, 1);
    dim3 block(threads_per_block, 1, 1);

    printf("    Blocks used: %d x %d = %d\n", dimx, dimy, dimx * dimy);

    printf("    Available registers per thread: %d (%d per block)\n", MAX_REGISTERS_PER_BLOCK / threads_per_block, MAX_REGISTERS_PER_BLOCK);

    int max_shared_memory;
    cudaDeviceGetAttribute(&max_shared_memory, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);

    #ifndef NO_SWIZZLE
    constexpr unsigned int shared_memory_used_A = shared_m * shared_k * sizeof(elmT) * NUM_STAGES;
    constexpr unsigned int shared_memory_used_B = shared_k * shared_n * sizeof(elmT) * NUM_STAGES;
    #else
    constexpr unsigned int shared_memory_used_A = shared_m * (shared_k + SHARED_PADDING) * sizeof(elmT) * NUM_STAGES;
    constexpr unsigned int shared_memory_used_B = shared_k * (shared_n + SHARED_PADDING) * sizeof(elmT) * NUM_STAGES;
    #endif

    constexpr unsigned int shared_memory_used = shared_memory_used_A + shared_memory_used_B;

    printf("    Shared memory used: %d/%d bytes (%.0f%%)\n", shared_memory_used, max_shared_memory, (float) shared_memory_used / max_shared_memory * 100);
    printf("    Shared memory used A: %d/%d bytes (%.0f%%)\n", shared_memory_used_A, max_shared_memory, (float) shared_memory_used_A / max_shared_memory * 100);
    printf("    Shared memory used B: %d/%d bytes (%.0f%%)\n", shared_memory_used_B, max_shared_memory, (float) shared_memory_used_B / max_shared_memory * 100);

    auto kernel = matMulTiledTensor<elmT, elmAccT, WMMA_M, WMMA_N, WMMA_K, FRAGS_M, FRAGS_N, FRAGS_K, WARP_TILES_M, WARP_TILES_N, WARP_TILES_K, BLOCK_TILES_M, BLOCK_TILES_N, threads_per_block, NUM_STAGES>;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_used);
//    cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
//    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

    TimeMeasurement t;

    t.start();
    for (int i = 0; i < n_runs; i++) {
//        TODO: fix requested amount of shared memory
        kernel<<<grid, block, shared_memory_used>>>(
            A_device, B_device, C_device, m, n, k
            );
    }
    cudaDeviceSynchronize();
    t.stop();

    // Check if kernel launch was successfull
    gpuAssert(cudaPeekAtLastError());
    return t.elapsed();
}



// TODO: remove this
template <typename elmT, typename elmAccT = elmT>
long int benchmark_cutlass_mmm_simple(int n_runs, elmT * A, elmT * B, elmAccT * C, int m, int n, int k);

// TODO: make general in element types?
template<>
long int benchmark_cutlass_mmm_simple<half_t, float>(int n_runs,
                                                     half_t * A, half_t * B, float * C,
                                                     int m, int n, int k)
{
    using namespace cute;

    using TA = half_t;
    using TB = half_t;
    using TC = float;

//    TODO: get as argument?
    auto alpha = Int<1>{};
    auto beta = Int<0>{};

    // Define shapes (dynamic)
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto prob_shape = make_shape(M, N, K);                   // (M, N, K)

    // Define strides (mixed)
    auto dA = make_stride(K, Int<1>{});                      // (dM, dK)
    auto dB = make_stride(Int<1>{}, N);                      // (dN, dK)
    auto dC = make_stride(N, Int<1>{});                      // (dM, dN)


    // Define mma tiles (static)
    TiledMMA tiled_mma = make_tiled_mma(
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>{},
        Layout<Shape<Int<BLOCK_TILES_M>,Int<BLOCK_TILES_N>,_1>>{},
        Tile<Int<BLOCK_TILES_M * WMMA_M>, Int<BLOCK_TILES_N * WMMA_N>, Int<WMMA_K>>{}
    );


    //    TODO: try more configs, pipelining, prefetched synchronous copies
    // Define shared memory layout (static)
    auto bM = Int<WMMA_M * FRAGS_M * WARP_TILES_M * BLOCK_TILES_M>{};
    auto bN = Int<WMMA_N * FRAGS_N * WARP_TILES_N * BLOCK_TILES_N>{};
    auto bK = Int<WMMA_K * FRAGS_K * WARP_TILES_K>{};
    auto cta_tiler = make_shape(bM, bN, bK);                 // (BLK_M, BLK_N, BLK_K)

    auto swizzle_layoutAtom_A =
            composition(
            Swizzle<3,3,3>{},
            Layout<
                    Shape < _8,_64>,
                    Stride<_64, _1>
            >{}
    );
    auto swizzle_layoutAtom_B =
            composition(
            Swizzle<3,3,3>{},
            Layout<
                    Shape <_64, _8>,
                    Stride< _1,_64>
            >{}
    );

    auto sA = tile_to_shape(swizzle_layoutAtom_A, make_shape(bM, bK));
    auto sB = tile_to_shape(swizzle_layoutAtom_B, make_shape(bN, bK));
    auto sC = make_layout(make_shape(bM, bN), LayoutRight{});

    // Define global->shared copy tiling (static)
#ifdef SYNC_CPY
    using ACopyOpGlobalShared = UniversalCopy<uint128_t>;
    using BCopyOpGlobalShared = UniversalCopy<uint128_t>;
#else
    using ACopyOpGlobalShared = SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>;
    using BCopyOpGlobalShared = SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>;
#endif

    TiledCopy copyA_global_shared = make_tiled_copy(Copy_Atom<ACopyOpGlobalShared, TA>{},
            Layout<
                    Shape<Int<BLOCK_TILES_M * BLOCK_TILES_N * WARP_SIZE / (WMMA_K * FRAGS_K * WARP_TILES_K / 8)>, Int<WMMA_K * FRAGS_K * WARP_TILES_K / 8>>,
                    Stride<Int<WMMA_K * FRAGS_K * WARP_TILES_K / 8>,_1>
            >{},
            Layout<Shape<_1,_8>>{}
    );

    TiledCopy copyB_global_shared = make_tiled_copy(Copy_Atom<BCopyOpGlobalShared, TB>{},
            Layout<
                    Shape<Int<WMMA_N * FRAGS_N * WARP_TILES_N * BLOCK_TILES_N / 8>, Int<BLOCK_TILES_M * BLOCK_TILES_N * WARP_SIZE / (WMMA_N * FRAGS_N * WARP_TILES_N * BLOCK_TILES_N / 8)>>,
                    Stride<_1, Int<WMMA_N * FRAGS_N * WARP_TILES_N * BLOCK_TILES_N / 8>>
            >{},
            Layout<Shape<_8,_1>>{}
    );


    // Define shared->register copy tiling (static)
#ifdef NO_LDSM
    using ACopyOpSharedRegisters = AutoVectorizingCopy;
    using BCopyOpSharedRegisters = AutoVectorizingCopy;
#else
    using ACopyOpSharedRegisters = SM75_U32x4_LDSM_N;
    using BCopyOpSharedRegisters = SM75_U16x8_LDSM_T;
#endif

    TiledCopy copyA_shared_registers = make_tiled_copy_A(Copy_Atom<ACopyOpSharedRegisters, TA>{}, tiled_mma);
    TiledCopy copyB_shared_registers = make_tiled_copy_B(Copy_Atom<BCopyOpSharedRegisters, TB>{}, tiled_mma);


    // Define kernel parameters
    auto kernel = gemm_simple<
            decltype(prob_shape), decltype(cta_tiler),
            TA, decltype(dA), decltype(sA), decltype(copyA_global_shared), decltype(copyA_shared_registers),
            TB, decltype(dB), decltype(sB), decltype(copyB_global_shared), decltype(copyB_shared_registers),
            TC, decltype(dC), decltype(sC), decltype(tiled_mma),
            decltype(alpha), decltype(beta)
    >;

    const uint32_t shared_memory_used = cosize_v<decltype(sA)> * sizeof(TA) + cosize_v<decltype(sB)> * sizeof(TB);
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_used);
    dim3 dimBlock(size(tiled_mma));
    dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));


//    print_latex(swizzle_layoutAtom_A);
//
//    print_latex(copyA_global_shared);
//
////    TODO: figure out how to best show swizzled copy
//    auto [layoutS_MN_global_shared, thrID_S_global_shared] = copyA_global_shared.get_layoutS_MN();
//    auto [layoutD_MN_global_shared, thrID_D_global_shared] = copyA_global_shared.get_layoutD_MN();
////    print_latex_copy(layoutS_MN_global_shared, thrID_S_global_shared, composition(sA, layoutD_MN_global_shared), thrID_D_global_shared);
////    print_latex_copy(layoutS_MN_global_shared, thrID_S_global_shared, sA, thrID_D_global_shared);
////    print_latex_copy(layoutS_MN_global_shared, thrID_S_global_shared, composition(layoutD_MN_global_shared, right_inverse(swizzle_layoutAtom_A).with_shape(layoutD_MN_global_shared.shape())), thrID_D_global_shared);
////    print_latex_copy(layoutS_MN_global_shared, thrID_S_global_shared, composition(right_inverse(tile_to_shape(swizzle_layoutAtom_A, Shape<_16, _64>{})), layoutD_MN_global_shared), thrID_D_global_shared);
////    print_latex_copy(layoutS_MN_global_shared, thrID_S_global_shared, composition(tile_to_shape(swizzle_layoutAtom_A, Shape<_16, _64>{}), layoutD_MN_global_shared), thrID_D_global_shared);
//    print_latex_copy(layoutS_MN_global_shared, thrID_S_global_shared, layoutD_MN_global_shared, thrID_D_global_shared, tile_to_shape(swizzle_layoutAtom_A, Shape<_16, _64>{}));
//
//    print_latex(copyA_shared_registers);
//
//    auto [layoutS_MN_shared_registers, thrID_S_shared_registers] = copyA_shared_registers.get_layoutS_MN();
//    auto [layoutD_MN_shared_registers, thrID_D_shared_registers] = copyA_shared_registers.get_layoutD_MN();
//    print_latex_copy(composition(sA, layoutS_MN_shared_registers), thrID_S_shared_registers, layoutD_MN_shared_registers, thrID_D_shared_registers);
//
//    print_latex(tiled_mma);


    // Launch kernel
    TimeMeasurement t;
    t.start();
    for (int i = 0; i < n_runs; i++) {
        kernel<<<dimGrid, dimBlock, shared_memory_used>>>(
            prob_shape,
            A, dA,
            B, dB,
            C, dC,
            alpha, beta
        );
    }
    cudaDeviceSynchronize();
    t.stop();

    // Check if kernel launch was successfull
    gpuAssert(cudaPeekAtLastError());
    return t.elapsed();
}


// TODO: generalize to any elm types
template <typename elmT, typename elmAccT = elmT>
long int benchmark_cute_mmm(int n_runs, elmT * A, elmT * B, elmAccT * C, int m, int n, int k);

template<>
long int benchmark_cute_mmm<half_t, float>(int n_runs, half_t * A, half_t * B, float * C, int m, int n, int k) {
    using namespace cute;

    using TA = half_t;
    using TB = half_t;
    using TC = float;

//    TODO: get as argument?
    auto alpha = Int<1>{};
    auto beta = Int<0>{};

    // Define shapes (dynamic)
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto prob_shape = make_shape(M, N, K);                   // (M, N, K)

    // Define strides (mixed)
    auto dA = make_stride(K, Int<1>{});                      // (dM, dK)
    auto dB = make_stride(Int<1>{}, N);                      // (dN, dK)
    auto dC = make_stride(N, Int<1>{});                      // (dM, dN)

    // Define mma tiles (static)
    TiledMMA tiled_mma = make_tiled_mma(
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>{},
        Layout<Shape<Int<BLOCK_TILES_M>,Int<BLOCK_TILES_N>,_1>>{},
        Tile<Int<BLOCK_TILES_M * WMMA_M>, Int<BLOCK_TILES_N * WMMA_N>, Int<WMMA_K>>{}
    );

    //    TODO: smarter way to calculate config from compiler defs
    // Define shared memory layout (static)
    auto bM = Int<WMMA_M * FRAGS_M * WARP_TILES_M * BLOCK_TILES_M>{};
    auto bN = Int<WMMA_N * FRAGS_N * WARP_TILES_N * BLOCK_TILES_N>{};
    auto bK = Int<WMMA_K * FRAGS_K * WARP_TILES_K>{};
    auto bP = Int<NUM_STAGES>{};
    auto cta_tiler = make_shape(bM, bN, bK);                 // (BLK_M, BLK_N, BLK_K)

    using SharedM = decltype(bM);
    using SharedN = decltype(bN);
    using SharedK = decltype(bK);

    using layoutAtom_A = Layout<
            // TODO: use min of shared_k and 64 instead of 64?
            Shape < _8,_64>,
            Stride<_64, _1>
    >;
    using layoutAtom_B = Layout<
            // TODO: use min of shared_n and 64 instead of 64?
            Shape <_64, _8>,
            Stride< _1,_64>
    >;

#ifdef NO_SWIZZLE
//    TODO: add padding?
    auto swizzle_layoutAtom_A = layoutAtom_A{};
    auto swizzle_layoutAtom_B = layoutAtom_B{};
#else
    auto swizzle_layoutAtom_A = composition(Swizzle<3,3,3>{}, layoutAtom_A{});
    auto swizzle_layoutAtom_B = composition(Swizzle<3,3,3>{}, layoutAtom_B{});
#endif

#ifdef SYNC_CPY
    auto sA = tile_to_shape(swizzle_layoutAtom_A, make_shape(bM, bK));
    auto sB = tile_to_shape(swizzle_layoutAtom_B, make_shape(bN, bK));

#ifdef NO_VECTORIZE
    using ACopyOpGlobalShared = UniversalCopy<half_t>;
    using BCopyOpGlobalShared = UniversalCopy<half_t>;

    const int elms_per_load = 1;
#else
    using ACopyOpGlobalShared = UniversalCopy<uint128_t>;
    using BCopyOpGlobalShared = UniversalCopy<uint128_t>;

    const int elms_per_load = 8;
#endif
#else

#if (NUM_STAGES == 1)
        auto sA = tile_to_shape(swizzle_layoutAtom_A, make_shape(bM, bK));
        auto sB = tile_to_shape(swizzle_layoutAtom_B, make_shape(bN, bK));
#else
        auto sA = tile_to_shape(swizzle_layoutAtom_A, make_shape(bM, bK, bP));
        auto sB = tile_to_shape(swizzle_layoutAtom_B, make_shape(bN, bK, bP));
#endif

#ifdef NO_VECTORIZE
    using ACopyOpGlobalShared = SM80_CP_ASYNC_CACHEALWAYS<uint32_t>;
    using BCopyOpGlobalShared = SM80_CP_ASYNC_CACHEALWAYS<uint32_t>;

    const int elms_per_load = 2;
#else
    using ACopyOpGlobalShared = SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>;
    using BCopyOpGlobalShared = SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>;

    const int elms_per_load = 8;
#endif
#endif

    auto sC = make_layout(make_shape(bM, bN), LayoutRight{});

    // Define global->shared copy tiling (static)
    TiledCopy copyA_global_shared = make_tiled_copy(Copy_Atom<ACopyOpGlobalShared, TA>{},
        Layout<
            Shape<Int<BLOCK_TILES_M * BLOCK_TILES_N * WARP_SIZE / (WMMA_K * FRAGS_K * WARP_TILES_K / elms_per_load)>, Int<WMMA_K * FRAGS_K * WARP_TILES_K / elms_per_load>>,
            Stride<Int<WMMA_K * FRAGS_K * WARP_TILES_K / elms_per_load>,_1>
        >{},
        Layout<Shape<_1,Int<elms_per_load>>>{}
    );

    TiledCopy copyB_global_shared = make_tiled_copy(Copy_Atom<BCopyOpGlobalShared, TB>{},
        Layout<
            Shape<Int<WMMA_N * FRAGS_N * WARP_TILES_N * BLOCK_TILES_N / elms_per_load>, Int<BLOCK_TILES_M * BLOCK_TILES_N * WARP_SIZE / (WMMA_N * FRAGS_N * WARP_TILES_N * BLOCK_TILES_N / elms_per_load)>>,
            Stride<_1, Int<WMMA_N * FRAGS_N * WARP_TILES_N * BLOCK_TILES_N / elms_per_load>>
        >{},
        Layout<Shape<Int<elms_per_load>,_1>>{}
    );

    // Define shared->register copy tiling (static)
#ifdef NO_LDSM
    using ACopyOpSharedRegisters = AutoVectorizingCopy;
    using BCopyOpSharedRegisters = AutoVectorizingCopy;
#else
    using ACopyOpSharedRegisters = SM75_U32x4_LDSM_N;
    using BCopyOpSharedRegisters = SM75_U16x8_LDSM_T;
#endif

    TiledCopy copyA_shared_registers = make_tiled_copy_A(Copy_Atom<ACopyOpSharedRegisters, TA>{}, tiled_mma);
    TiledCopy copyB_shared_registers = make_tiled_copy_B(Copy_Atom<BCopyOpSharedRegisters, TB>{}, tiled_mma);

    // Define kernel parameters
#if (NUM_STAGES == 1)
    auto kernel = gemm_simple<
            decltype(prob_shape), decltype(cta_tiler),
            TA, decltype(dA), decltype(sA), decltype(copyA_global_shared), decltype(copyA_shared_registers),
            TB, decltype(dB), decltype(sB), decltype(copyB_global_shared), decltype(copyB_shared_registers),
            TC, decltype(dC), decltype(sC), decltype(tiled_mma),
            decltype(alpha), decltype(beta)
    >;
#else
#ifdef SYNC_CPY
    static_assert(NUM_STAGES == 2, "NUM_STAGES must be 2 for gemm_sync_cpy");

    auto kernel = gemm_sync_cpy<
            decltype(prob_shape), decltype(cta_tiler),
            TA, decltype(dA), decltype(sA), decltype(copyA_global_shared), decltype(copyA_shared_registers),
            TB, decltype(dB), decltype(sB), decltype(copyB_global_shared), decltype(copyB_shared_registers),
            TC, decltype(dC), decltype(sC), decltype(tiled_mma),
            decltype(alpha), decltype(beta)
    >;
#else
    auto kernel = gemm_pipelined<
            decltype(prob_shape), decltype(cta_tiler),
            TA, decltype(dA), decltype(sA), decltype(copyA_global_shared), decltype(copyA_shared_registers),
            TB, decltype(dB), decltype(sB), decltype(copyB_global_shared), decltype(copyB_shared_registers),
            TC, decltype(dC), decltype(sC), decltype(tiled_mma),
            decltype(alpha), decltype(beta),
            NUM_STAGES
    >;
#endif
#endif

    const uint32_t shared_memory_used = cosize_v<decltype(sA)> * sizeof(TA) + cosize_v<decltype(sB)> * sizeof(TB);
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_used);
    dim3 dimBlock(size(tiled_mma));
    dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));

    printf("Used shared memory: %d, threads: %d\n", shared_memory_used, (int)(size(tiled_mma)));

    // Launch kernel
    TimeMeasurement t;
    t.start();
    for (int i = 0; i < n_runs; i++) {
        kernel<<<dimGrid, dimBlock, shared_memory_used>>>(
                prob_shape,
                A, dA,
                B, dB,
                C, dC,
                alpha, beta
        );
    }
    cudaDeviceSynchronize();
    t.stop();

    // Check if kernel launch was successfull
    gpuAssert(cudaPeekAtLastError());
    return t.elapsed();
}

template <typename elmT, typename elmAccT = elmT>
long int benchmark_cute_attention_like(unsigned int n_runs, elmT * As, elmT * Bss, elmAccT * Cs, unsigned int batches, unsigned int reuse);

template<>
long int benchmark_cute_attention_like<half_t, float>(unsigned int n_runs, half_t * As, half_t * Bss, float * Cs, unsigned int batches, unsigned int reuse) {
//        constexpr unsigned int shared_m = WMMA_M * FRAGS_M * WARP_TILES_M * BLOCK_TILES_M;
//        constexpr unsigned int shared_n = WMMA_N * FRAGS_N * WARP_TILES_N * BLOCK_TILES_N;
//        constexpr unsigned int shared_k = WMMA_K * FRAGS_K * WARP_TILES_K;

    using namespace cute;

    using TA = half_t;
    using TB = half_t;
    using TC = float;

//    TODO: get as argument?
    auto alpha = Int<1>{};
    auto beta = Int<0>{};

    // Define mma tiles (static)
    TiledMMA tiled_mma = make_tiled_mma(
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>{},
        Layout<Shape<Int<BLOCK_TILES_M>,Int<BLOCK_TILES_N>,_1>>{},
        Tile<Int<BLOCK_TILES_M * WMMA_M>, Int<BLOCK_TILES_N * WMMA_N>, Int<WMMA_K>>{}
    );

    //    TODO: smarter way to calculate config from compiler defs
    // Define shared memory layout (static)
    auto bM = Int<WMMA_M * FRAGS_M * WARP_TILES_M * BLOCK_TILES_M>{};
    auto bN = Int<WMMA_N * FRAGS_N * WARP_TILES_N * BLOCK_TILES_N>{};
    auto bK = Int<WMMA_K * FRAGS_K * WARP_TILES_K>{};
    auto bP = Int<NUM_STAGES>{};

    using SharedM = decltype(bM);
    using SharedN = decltype(bN);
    using SharedK = decltype(bK);

    auto layoutAs = make_layout(make_shape(bM, bK, batches), make_stride(bK, Int<1>{}, bM * bK));
    auto layoutBss = make_layout(make_shape(bN, bK, batches, reuse), make_stride(Int<1>{}, bN, bN * bK * reuse, bN * bK));
    auto layoutCs = make_layout(make_shape(bM, bN, batches), make_stride(bN, Int<1>{}, bM * bN));


    using layoutAtom_A = Layout<
            // TODO: use min of shared_k and 64 instead of 64?
            Shape < _8,_64>,
            Stride<_64, _1>
    >;
    using layoutAtom_B = Layout<
            // TODO: use min of shared_n and 64 instead of 64?
            Shape <_64, _8>,
            Stride< _1,_64>
    >;

#ifdef NO_SWIZZLE
    //    TODO: add padding?
    auto swizzle_layoutAtom_A = layoutAtom_A{};
    auto swizzle_layoutAtom_B = layoutAtom_B{};
#else
    auto swizzle_layoutAtom_A = composition(Swizzle<3,3,3>{}, layoutAtom_A{});
    auto swizzle_layoutAtom_B = composition(Swizzle<3,3,3>{}, layoutAtom_B{});
#endif

#ifdef SYNC_CPY
    auto sA = tile_to_shape(swizzle_layoutAtom_A, make_shape(bM, bK));
    auto sB = tile_to_shape(swizzle_layoutAtom_B, make_shape(bN, bK));

#ifdef NO_VECTORIZE
    using ACopyOpGlobalShared = UniversalCopy<half_t>;
    using BCopyOpGlobalShared = UniversalCopy<half_t>;

    const int elms_per_load = 1;
#else
    using ACopyOpGlobalShared = UniversalCopy<uint128_t>;
    using BCopyOpGlobalShared = UniversalCopy<uint128_t>;

    const int elms_per_load = 8;
#endif
#else

#if (NUM_STAGES == 1)
    auto sA = tile_to_shape(swizzle_layoutAtom_A, make_shape(bM, bK));
    auto sB = tile_to_shape(swizzle_layoutAtom_B, make_shape(bN, bK));
#else
    auto sA = tile_to_shape(swizzle_layoutAtom_A, make_shape(bM, bK, bP));
    auto sB = tile_to_shape(swizzle_layoutAtom_B, make_shape(bN, bK, bP));
#endif

#ifdef NO_VECTORIZE
    using ACopyOpGlobalShared = SM80_CP_ASYNC_CACHEALWAYS<uint32_t>;
    using BCopyOpGlobalShared = SM80_CP_ASYNC_CACHEALWAYS<uint32_t>;

    const int elms_per_load = 2;
#else
    using ACopyOpGlobalShared = SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>;
    using BCopyOpGlobalShared = SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>;

    const int elms_per_load = 8;
#endif
#endif

    auto sC = make_layout(make_shape(bM, bN), LayoutRight{});

    // Define global->shared copy tiling (static)
    TiledCopy copyA_global_shared = make_tiled_copy(Copy_Atom<ACopyOpGlobalShared, TA>{},
        Layout<
            Shape<Int<BLOCK_TILES_M * BLOCK_TILES_N * WARP_SIZE / (WMMA_K * FRAGS_K * WARP_TILES_K / elms_per_load)>, Int<WMMA_K * FRAGS_K * WARP_TILES_K / elms_per_load>>,
            Stride<Int<WMMA_K * FRAGS_K * WARP_TILES_K / elms_per_load>,_1>
        >{},
        Layout<Shape<_1,Int<elms_per_load>>>{}
    );

    TiledCopy copyB_global_shared = make_tiled_copy(Copy_Atom<BCopyOpGlobalShared, TB>{},
        Layout<
            Shape<Int<WMMA_N * FRAGS_N * WARP_TILES_N * BLOCK_TILES_N / elms_per_load>, Int<BLOCK_TILES_M * BLOCK_TILES_N * WARP_SIZE / (WMMA_N * FRAGS_N * WARP_TILES_N * BLOCK_TILES_N / elms_per_load)>>,
            Stride<_1, Int<WMMA_N * FRAGS_N * WARP_TILES_N * BLOCK_TILES_N / elms_per_load>>
        >{},
    Layout<Shape<Int<elms_per_load>,_1>>{}
    );

    // Define shared->register copy tiling (static)
#ifdef NO_LDSM
    using ACopyOpSharedRegisters = AutoVectorizingCopy;
    using BCopyOpSharedRegisters = AutoVectorizingCopy;
#else
    using ACopyOpSharedRegisters = SM75_U32x4_LDSM_N;
    using BCopyOpSharedRegisters = SM75_U16x8_LDSM_T;
#endif

    TiledCopy copyA_shared_registers = make_tiled_copy_A(Copy_Atom<ACopyOpSharedRegisters, TA>{}, tiled_mma);
    TiledCopy copyB_shared_registers = make_tiled_copy_B(Copy_Atom<BCopyOpSharedRegisters, TB>{}, tiled_mma);

    // Define kernel parameters
//    TODO: allow setting num_stages?
//#if (NUM_STAGES == 1)
//    auto kernel = gemm_simple<
//            decltype(prob_shape), decltype(cta_tiler),
//            TA, decltype(dA), decltype(sA), decltype(copyA_global_shared), decltype(copyA_shared_registers),
//            TB, decltype(dB), decltype(sB), decltype(copyB_global_shared), decltype(copyB_shared_registers),
//            TC, decltype(dC), decltype(sC), decltype(tiled_mma),
//            decltype(alpha), decltype(beta)
//    >;
//#else
//#ifdef SYNC_CPY
//    static_assert(NUM_STAGES == 2, "NUM_STAGES must be 2 for gemm_sync_cpy");
//
//    auto kernel = gemm_sync_cpy<
//            decltype(prob_shape), decltype(cta_tiler),
//            TA, decltype(dA), decltype(sA), decltype(copyA_global_shared), decltype(copyA_shared_registers),
//            TB, decltype(dB), decltype(sB), decltype(copyB_global_shared), decltype(copyB_shared_registers),
//            TC, decltype(dC), decltype(sC), decltype(tiled_mma),
//            decltype(alpha), decltype(beta)
//    >;
//#else
//    auto kernel = gemm_pipelined<
//            decltype(prob_shape), decltype(cta_tiler),
//            TA, decltype(dA), decltype(sA), decltype(copyA_global_shared), decltype(copyA_shared_registers),
//            TB, decltype(dB), decltype(sB), decltype(copyB_global_shared), decltype(copyB_shared_registers),
//            TC, decltype(dC), decltype(sC), decltype(tiled_mma),
//            decltype(alpha), decltype(beta),
//            NUM_STAGES
//    >;
//#endif
//#endif
    #ifdef ATTENTION_LIKE
    static_assert(NUM_STAGES == 1, "NUM_STAGES must be 1 for attention_like");

    auto kernel = attention_like_simple<
        TA, decltype(layoutAs), decltype(sA), decltype(copyA_global_shared), decltype(copyA_shared_registers),
        TB, decltype(layoutBss), decltype(sB), decltype(copyB_global_shared), decltype(copyB_shared_registers),
        TC, decltype(layoutCs), decltype(sC), decltype(tiled_mma),
        decltype(alpha), decltype(beta)
    >;

    const uint32_t shared_memory_used = cosize_v<decltype(sA)> * sizeof(TA) + cosize_v<decltype(sB)> * sizeof(TB);
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_used);
    dim3 dimBlock(size(tiled_mma));
    dim3 dimGrid(batches);

    printf("Used shared memory: %d, threads: %d\n", shared_memory_used, (int)(size(tiled_mma)));

    // Launch kernel
    TimeMeasurement t;
    t.start();
    for (int i = 0; i < n_runs; i++) {
        kernel<<<dimGrid, dimBlock, shared_memory_used>>>(
                As, layoutAs,
                Bss, layoutBss,
                Cs, layoutCs,
                alpha, beta
        );
    }
    cudaDeviceSynchronize();
    t.stop();

    // Check if kernel launch was successfull
    gpuAssert(cudaPeekAtLastError());
    return t.elapsed();
    #else
    return 0;
    #endif
}


template <typename elmT, typename elmAccT = elmT>
long int benchmark_cute_default(int n_runs, elmT * A, elmT * B, elmAccT * C, int m, int n, int k);

template<>
long int benchmark_cute_default<half_t, float>(int n_runs, half_t * A, half_t * B, float * C, int m, int n, int k) {
    using namespace cute;

    // TODO: remove?
    const auto alpha = static_cast<float>(1);
    const auto beta  = static_cast<float>(0);

    // Define shapes (dynamic)
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

    // Define strides (mixed)
    auto dA = make_stride(K, Int<1>{});                      // (dM, dK)
    auto dB = make_stride(Int<1>{}, N);                      // (dN, dK)
    auto dC = make_stride(N, Int<1>{});                      // (dM, dN)

//    // Define CTA tile sizes (static)
////    TODO: get from calculation, use 128 rather than 64
//    auto bM = Int<64>{};
//    auto bN = Int<64>{};
//    auto bK = Int<32>{};
//    auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
//    auto bP = Int<3>{};  // Pipeline
//
//    auto sA_buffer = make_layout(make_shape(bM, bK), make_stride(bK + Int<8>{}, Int<1>{}));
//    auto sB_buffer = make_layout(make_shape(bN, bK), make_stride(Int<1>{}, bN + Int<8>{}));
//
//    // Define the smem layouts (static)
////    auto sA = make_layout(make_shape(bM, bK, bP), make_stride(bK, Int<1>{}));
////    auto sB = make_layout(make_shape(bK, bN, bP), LayoutRight{});
//    auto sA = tile_to_shape(sA_buffer, make_shape(bM, bK, bP));
//    auto sB = tile_to_shape(sB_buffer, make_shape(bN, bK, bP));
//    auto sC = make_layout(make_shape(bM, bN), LayoutRight{});

    using ALayout = decltype(dA);
    using BLayout = decltype(dB);
    using CLayout = decltype(dC);

    // TODO: set
//    using ThreadBlockSize = _128;
//    using TiledMma = ;

    using CopyMaxVecBits = _128;
    using TA = half_t;
    using TB = half_t;
    using TC = float;
    using Alpha = decltype(alpha);
    using Beta = decltype(beta);

//    auto kernel = cooperative_gemm_kernel<
//            SMemALayout, SMemBLayout, SMemCLayout,
//            SmemCopyOpA, SmemCopyOpB, SmemCopyOpC,
//            ThreadBlockSize, TiledMma, CopyMaxVecBits,
//            TA, TB, TC, Alpha, Beta,
//            ALayout, BLayout, CLayout
//    >;

//    TODO: use?
//    dim3 dimBlock(size(mmaC));
//    dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));

//    constexpr uint32_t copy_max_vec_bytes = CopyMaxVecBits / 8;
//    const size_t shared_memory_size = round_up(sizeof(TA) * h_a.size(), copy_max_vec_bytes)
//                                      + round_up(sizeof(TB) * h_b.size(), copy_max_vec_bytes)
//                                      +         (sizeof(TC) * h_c.size());
//    ASSERT_EQ(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_memory_size)), 0);

    TimeMeasurement t;
    t.start();
    for (int i = 0; i < n_runs; i++) {
//        TODO: set grid size
//        kernel<<<1, ThreadBlockSize, shared_memory_size>>>(
//                thrust::raw_pointer_cast(d_a.data()),
//                thrust::raw_pointer_cast(d_b.data()),
//                thrust::raw_pointer_cast(d_c.data()),
//                thrust::raw_pointer_cast(d_c_out.data()),
//                alpha,
//                beta,
//                a_load_transform,
//                b_load_transform,
//                c_load_transform,
//                c_store_transform
//        );
    }
    cudaDeviceSynchronize();
    t.stop();

    // Check if kernel launch was successfull
    gpuAssert(cudaPeekAtLastError());
    return t.elapsed();
}

template <cutlass_version cutlass_mmm, typename elmT, typename elmAccT = elmT>
long int benchmark_cutlass_mmm(int n_runs, elmT * A, elmT * B, elmAccT * C, int m, int n, int k)
{
    switch (cutlass_mmm)
    {
    case DEFAULT : 
        return cutlass_default_mmm(n_runs, A, B, C, m, n, k);     
    case PADDING:
        return cutlass_spadding_mmm(n_runs, A, B, C, m, n, k);
    case VECTORIZED : 
        return cutlass_vectorized_mmm(n_runs, A, B, C, m, n, k);
    default: return -1;
    }
}

// TODO: generalize to any elm types
template <typename elmT, typename elmAccT = elmT>
long int benchmark_cutlass_default(int n_runs, elmT * A, elmT * B, elmAccT * C, int m, int n, int k);

template<>
long int benchmark_cutlass_default<half_t, float>(int n_runs, half_t * A, half_t * B, float * C, int m, int n, int k) {
    using ElementA_ = half_t;
    using ElementB_ = half_t;
    using ElementC_ = float;
    using ElementAccumulator_ = float;
    using OperatorClass_ = cutlass::arch::OpClassTensorOp;
    using ArchTag_ = cutlass::arch::Sm80;

    using GemmConfiguration = cutlass::gemm::device::DefaultGemmConfiguration<
            OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
            ElementAccumulator_
    >;

    using CutlassGemm = cutlass::gemm::device::Gemm<
            ElementA_,        // Data-type of A matrix
            cutlass::layout::RowMajor,  // Layout of A matrix
            ElementB_,        // Data-type of B matrix
            cutlass::layout::RowMajor,  // Layout of B matrix
            ElementC_,        // Data-type of C matrix
            cutlass::layout::RowMajor,  // Layout of C matrix
            ElementAccumulator_,
            OperatorClass_,
            ArchTag_,
            GemmConfiguration::ThreadblockShape,
            GemmConfiguration::WarpShape,
            GemmConfiguration::InstructionShape,
            GemmConfiguration::EpilogueOutputOp,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
            GemmConfiguration::kStages
    >;

    CutlassGemm gemm_operator;

    CutlassGemm::Arguments args({m , n, k},  // Gemm Problem dimensions
                                {A, k},    // Tensor-ref for source matrix A
                                {B, n},    // Tensor-ref for source matrix B
                                {C, n},    // Tensor-ref for source matrix C
                                {C, n},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                {1, 0}     // Scalars used in the Epilogue
    );

    TimeMeasurement t;
    t.start();
    for (int i = 0; i < n_runs; i++) {
        gemm_operator(args);
    }
    cudaDeviceSynchronize();
    t.stop();

    // Check if kernel launch was successfull
    gpuAssert(cudaPeekAtLastError());
    return t.elapsed();
}


template <typename elmT, typename elmAccT = elmT>
long int benchmark_cutlass_custom(int n_runs, elmT * A, elmT * B, elmAccT * C, int m, int n, int k);

//template<>
//long int benchmark_cutlass_custom<half_t, float>(int n_runs, half_t * A, half_t * B, float * C, int m, int n, int k) {
//    using namespace cute;
//
//    // Define shapes (dynamic)
//    auto M = int(m);
//    auto N = int(n);
//    auto K = int(k);
//    auto prob_shape = make_shape(M, N, K);                     // (M, N, K)
//
//    // Define strides (mixed)
////    auto dA = make_stride(K, Int<1>{});                      // (dM, dK)
////    auto dB = make_stride(Int<1>{}, N);                      // (dN, dK)
////    auto dC = make_stride(N, Int<1>{});                      // (dM, dN)
//
//    using ElementA = half_t;
//    using LayoutA = cutlass::layout::RowMajor;
//    const int AlignmentA = 128 / sizeof_bits<ElementA>::value;
//    using ElementB = half_t;
//    using LayoutB = cutlass::layout::RowMajor;
//    const int AlignmentB = 128 / sizeof_bits<ElementB>::value;
//
//    using ElementC = float;
//    using LayoutC = cutlass::layout::RowMajor;
//    using ElementAccumulator = float;
//
//    using OperatorClass = cutlass::arch::OpClassTensorOp;
//    using ArchTag = cutlass::arch::Sm80;
//
////    using GemmConfiguration = cutlass::gemm::device::DefaultGemmConfiguration<
////            OperatorClass, ArchTag, ElementA, ElementB, ElementC,
////            ElementAccumulator
////    >;
//
//    // Step 1: Generate the required collective layer mainloop specialization
//    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
//            ArchTag, OperatorClass,
//            ElementA, LayoutA, AlignmentA,
//            ElementB, LayoutB, AlignmentB,
//            ElementAccumulator,
//            Shape<_64, _64, _64>, Shape<_1, _1, _1>,
//            cutlass::gemm::collective::StageCountAuto,
//            cutlass::gemm::collective::KernelScheduleAuto
//    >::CollectiveOp;
//
//// Step 2: Specify the collective layer epilogue type
//    using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
//            cutlass::gemm::TagToStrideC_t<LayoutC>,
//            cutlass::gemm::TagToStrideC_t<LayoutC>,
//            cutlass::epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>>;
//
//// Step 3: Compose the mainloop and epilogue together at the kernel layer
//    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
//            cute::Shape<int,int,int,int>, // ProblemShape [M,N,K,L]
//            CollectiveMainloop,
//            CollectiveEpilogue
//    >;
//
//// Step 4: Wrap up the kernel::GemmUniversal kernel class
//// with the device adapter to obtain a host-side handle to the kernel
//    using CutlassGemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
//
//    CutlassGemm gemm_operator;
//
//    CutlassGemm::Arguments args({m , n, k},  // Gemm Problem dimensions
//                                {A, k},    // Tensor-ref for source matrix A
//                                {B, n},    // Tensor-ref for source matrix B
//                                {C, n},    // Tensor-ref for source matrix C
//                                {C, n},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
//                                {1, 0}     // Scalars used in the Epilogue
//    );
//
//    TimeMeasurement t;
//    t.start();
//    for (int i = 0; i < n_runs; i++) {
//        gemm_operator(args);
//    }
//    cudaDeviceSynchronize();
//    t.stop();
//
//    // Check if kernel launch was successfull
//    gpuAssert(cudaPeekAtLastError());
//    return t.elapsed();
//}


template <typename elmT, typename elmAccT = elmT>
unsigned benchmark_naive_tensor_mmm(
    unsigned n_runs,
    elmT *A_device,
    elmT *B_device,
    elmAccT *ResMat_device,
    int m,
    int n,
    int k)
{
    constexpr int block_tiles_m = 8;
    constexpr int block_tiles_n = 4;
    constexpr int block_tiles_k = 4;
    constexpr int wmma_n = 16;
    constexpr int wmma_m = 16;
    constexpr int wmma_k = 16;


    // Let block work on block_tiles * wmma elements.
    // there are n elements on the x direction and we know each thread works on block_tiles_n
    int dimx = ceil(((float) n)/(wmma_n * block_tiles_n));
    int dimy = ceil( ((float) m)/(wmma_m * block_tiles_m));
    dim3 grid(dimx, dimy, 1);
    // dim3 block(threads_per_block, 1, 1); // 1D block of 256 elements
    /* Okay so what do we want? Each mm will be done by the entire warp and works warp level.
    So whatever we want to tile for should be multiple of the warp size.
    Here we say that the block should compute block_tiles_m x block_tiles_n tensor mm.

    This also works for the grid specification, since we tile so that each warp computes
    a wmma_m x wmma_n result, and we use block_tiles_m x block_tiles_n warps in the block.
    */
    dim3 block(block_tiles_n * WARP_SIZE, block_tiles_m, 1);

    TimeMeasurement t;

    t.start();
    for (int i = 0; i < n_runs; i++) {
        matMulTiledTensorNaive<
        elmAccT, elmT, wmma_m, wmma_n, wmma_k, block_tiles_m, block_tiles_n, block_tiles_k>
        <<<grid, block>>>(A_device, B_device, ResMat_device, m, n, n);
    }
    cudaDeviceSynchronize();
    t.stop();
    // Check if kernel launch was successfull
    gpuAssert(cudaPeekAtLastError());

    return t.elapsed();
}


template <typename elmT, typename elmAccT>
long int benchmark_tiled_mmm(
    int n_runs,
    elmT *A_device,
    elmT *B_device,
    elmAccT *C_device,
    int m,
    int n,
    int k)
{
    constexpr int tile_size = 16;
    constexpr int reg_size = 5;

    int dimy = ceil( ((float) n)/(tile_size * reg_size));
    int dimx = ceil( ((float) m)/(tile_size * reg_size));
    TimeMeasurement t;
    dim3 grid(dimx, dimy, 1);
    dim3 block(16, 16, 1);

    t.start();
    for (int i = 0; i < n_runs; i++) {
        matMulTiled<elmT, elmAccT, tile_size, reg_size, tile_size, reg_size, tile_size><<<grid, block>>>(
            A_device, B_device, C_device, m, n, k);
    }
    cudaDeviceSynchronize();
    t.stop();
    // Check if kernel launch was successfull
    gpuAssert(cudaPeekAtLastError());
    return t.elapsed();
}


template <typename elmT, typename elmAccT>
cublasStatus_t cublas_wrapper(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const elmAccT *alpha,
    const elmT *A, int lda,
    const elmT *B, int ldb,
    const elmAccT *beta,
    elmAccT *C, int ldc
    );

template <>
cublasStatus_t cublas_wrapper<half_t, half_t>(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,
        const half_t *alpha,
        const half_t *A, int lda,
        const half_t *B, int ldb,
        const half_t *beta,
        half_t *C, int ldc
) {
    return cublasGemmEx(
        handle,
        transa, transb,
        m, n, k,
        alpha,
        A, CUDA_R_16F, lda,
        B, CUDA_R_16F, ldb,
        beta,
        C, CUDA_R_16F, ldc,
        CUDA_R_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );
}

template <>
cublasStatus_t cublas_wrapper<float, float>(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const float *A, int lda,
    const float *B, int ldb,
    const float *beta,
    float *C, int ldc
    ) {
    return cublasGemmEx(
        handle,
        transa, transb,
        m, n, k,
        alpha,
        A, CUDA_R_32F, lda,
        B, CUDA_R_32F, ldb,
        beta,
        C, CUDA_R_32F, ldc,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );
}


template <>
cublasStatus_t cublas_wrapper<half_t, float>(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,
        const float *alpha,
        const half_t *A, int lda,
        const half_t *B, int ldb,
        const float *beta,
        float *C, int ldc
) {
    return cublasGemmEx(
        handle,
        transa, transb,
        m, n, k,
        alpha,
        A, CUDA_R_16F, lda,
        B, CUDA_R_16F, ldb,
        beta,
        C, CUDA_R_32F, ldc,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );
}


template <typename elmT, typename elmAccT>
long int benchmark_cublas(
    int n_runs,
    elmT *A_device,
    elmT *B_device,
    elmAccT *C_device,
    int m,
    int n,
    int k)
{
    TimeMeasurement t;

    cublasHandle_t handle;
    cublasStatus_t stat;
    stat = cublasCreate(&handle);
    elmAccT alpha = (elmAccT) 1.0;
    elmAccT beta = (elmAccT) 0.0;
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    t.start();
    for (int i = 0; i < n_runs; i++) {
        stat = cublas_wrapper<elmT, elmAccT>(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
            &alpha,
            // Cublas uses column major, so we need to swap A and B, since B^T @ A^T = (A @ B)^T = C^T
            B_device, n,
            A_device, k,
            &beta,
            C_device, n
            );
    }
    cudaDeviceSynchronize();
    t.stop();

    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS error\n");
        printf("%s\n", cublasGetStatusName(stat));
        printf("%s\n", cublasGetStatusString(stat));
        exit(1);
    }

    // Check if kernel launch was successfull
    gpuAssert(cudaPeekAtLastError());
    return t.elapsed();
}


// Expects A to have shape K x K and B to have K x N
template <typename elmT, typename elmAccT, int MatDim, mm_kernel kernel_type>
void run_mmm_kernel(
    int n_runs,
    int m,
    int n,
    int k,
    RandomMatrix<elmT, MatDim> &A,
    RandomMatrix<elmT, MatDim> &B,
    RandomMatrix<elmAccT, MatDim> &C)
{
    double total_ops = 2.0f * n * k * m;

    auto A_device = A.to_gpu();
    auto B_device = B.to_gpu();

    auto C_device = C.to_gpu();
    long int total_elapsed;

    if constexpr (kernel_type == mm_kernel::tensor_optimized) {
        total_elapsed = benchmark_optimized_tensor_mmm<elmT, elmAccT>(
            n_runs, A_device, B_device, C_device, m, n, k
            );
    }
    else if constexpr (kernel_type == mm_kernel::tensor_naive) {
        total_elapsed = benchmark_naive_tensor_mmm<elmT, elmAccT>(
            n_runs, A_device, B_device, C_device, m, n, k
            );
    }
    else if constexpr (kernel_type == mm_kernel::cublas) {
        total_elapsed = benchmark_cublas<elmT, elmAccT>(
            n_runs, A_device, B_device, C_device, m, n, k
            );
    }
    else if constexpr (kernel_type == mm_kernel::cute_mm) {
        total_elapsed = benchmark_cute_mmm(
                n_runs, A_device, B_device, C_device, m, n, k
        );
    } else if constexpr (kernel_type == mm_kernel::cutlass_default) {
        total_elapsed = benchmark_cutlass_default(
                n_runs, A_device, B_device, C_device, m, n, k
        );
    } else if constexpr (kernel_type == mm_kernel::cutlass_custom) {
        total_elapsed = benchmark_cutlass_custom(
                n_runs, A_device, B_device, C_device, m, n, k
        );
    } else if constexpr (kernel_type == mm_kernel::cutlass_simple) {
        total_elapsed = benchmark_cutlass_mmm_simple(
                n_runs, A_device, B_device, C_device, m, n, k
        );
    } else {
        total_elapsed = benchmark_tiled_mmm<elmT, elmAccT>(
            n_runs, A_device, B_device, C_device, m, n, k
            );
    }

    cudaMemcpy(C.to_cpu(), C_device, C.flatSize() * sizeof(elmAccT), cudaMemcpyDeviceToHost);


    if (!total_elapsed) {
        printf("Kernel launch failed\n");
        memset(C.to_cpu(), 0, m * n);
    } else {
        printGFlops(total_elapsed, total_ops * n_runs);
        printf("Average Time elapsed: %ld ms\n", total_elapsed / n_runs);
    }
}


// Expects A to have shape K x K and B to have K x N
template <typename elmT, typename accT, int MatDim, mm_kernel kernel_type, bool validate>
void benchmark_kernel(
    int n_runs,
    int m,
    int n,
    int k,
    RandomMatrix<elmT, MatDim> &A,
    RandomMatrix<elmT, MatDim> &B,
    RandomMatrix<accT, MatDim> &C,
    RandomMatrix<accT, MatDim> &C_target,
    std::string kernel_name
    ) {
    C.fill_zeros(m, n);

    std::cout << "-----" << std::endl;
    std::cout << "Running " << kernel_name << std::endl;
    std::cout << "Dry run" << std::endl;
    run_mmm_kernel<elmT, accT, MatDim, kernel_type>(
        1, m, n, k, A, B, C
        );

    RandomMatrix<accT, MatDim> C_actual;

    if (n_runs > 0) {
        std::cout << "Average run after: " << n_runs << " runs"<< std::endl;
        run_mmm_kernel<elmT, accT, MatDim, kernel_type>(
                n_runs, m, n, k, A, B, C
        );
        std::cout << "-----" << std::endl;
    }

    if constexpr (validate)
    {
        C_actual.fill_from(C, m, n);

        Validator<accT> validator(C_target.to_cpu(), C_actual.to_cpu(), m * n);
        // validator.setEps(0.000005); // original used by cosmin
        validator.setEps(0.0005);

        validator.validate();
    }
}


void benchmark_attention_like(
        unsigned int n_runs,
        unsigned int batches,
        unsigned int reuse
    ) {
    constexpr unsigned int shared_m = WMMA_M * FRAGS_M * WARP_TILES_M * BLOCK_TILES_M;
    constexpr unsigned int shared_n = WMMA_N * FRAGS_N * WARP_TILES_N * BLOCK_TILES_N;
    constexpr unsigned int shared_k = WMMA_K * FRAGS_K * WARP_TILES_K;

    RandomMatrix<element_type, 3> As;
    RandomMatrix<element_type, 4> Bss;
    RandomMatrix<acc_type, 3> Cs;

    As.fill_rand<float_range>(shared_m, shared_k, batches);
    Bss.fill_rand<float_range>(shared_k, shared_n, batches, reuse);
    Cs.fill_zeros(shared_m, shared_n, batches);

//    TODO: validation

    unsigned long total_ops = 2 * (unsigned long)batches * (unsigned long)reuse * shared_m * shared_n * shared_k;

    auto As_device = As.to_gpu();
    auto Bss_device = Bss.to_gpu();
    auto Cs_device = Cs.to_gpu();
    long int total_elapsed;

    std::cout << "-----" << std::endl;
    printf("Running attention like of size %d x %d x %d x %d x %d\n", batches, reuse, shared_m, shared_n, shared_k);
    std::cout << "Dry run" << std::endl;

    total_elapsed = benchmark_cute_attention_like(
            1, As_device, Bss_device, Cs_device, batches, reuse
    );

    if (!total_elapsed) {
        printf("Kernel launch failed\n");
//        memset(C.to_cpu(), 0, m * n);
    } else {
        printGFlops(total_elapsed, total_ops);
        printf("Average Time elapsed: %ld ms\n", total_elapsed);
    }


    if (n_runs > 0) {
        std::cout << "Average run after: " << n_runs << " runs"<< std::endl;
        total_elapsed = benchmark_cute_attention_like(
                n_runs, As_device, Bss_device, Cs_device, batches, reuse
        );

        if (!total_elapsed) {
            printf("Kernel launch failed\n");
//        memset(C.to_cpu(), 0, m * n);
        } else {
            printGFlops(total_elapsed, total_ops * n_runs);
            printf("Average Time elapsed: %ld ms\n", total_elapsed / n_runs);
        }
        std::cout << "-----" << std::endl;
    }

    cudaFree(As.to_gpu());
    cudaFree(Bss.to_gpu());
    cudaFree(Cs.to_gpu());
}


int main(int argc, char * argv[])
{
    int m = 16 * 256;
    int n = 16 * 256;
    int k = 16 * 256;

    int n_runs = 10;

    if (argc >= 2)
    {
        n_runs = atoi(argv[1]);
    }
    if (argc == 3)
    {
        int input_int = atoi(argv[2]);
        m = input_int;
        n = input_int;
        k = input_int;
    } else if (argc == 4)
    {
        n_runs = 10;
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        k = atoi(argv[3]);
    } else if (argc == 5)
    {
        n_runs = atoi(argv[1]);
        m = atoi(argv[2]);
        n = atoi(argv[3]);
        k = atoi(argv[4]);
    }


    TimeMeasurement t;

#ifdef ATTENTION_LIKE
    benchmark_attention_like(
        n_runs,
        m,
        n
    );
#else
    // Define matrices
    RandomMatrix<element_type, 2> A;
    RandomMatrix<element_type, 2> B;
    RandomMatrix<acc_type, 2> A_accT;
    RandomMatrix<acc_type, 2> B_accT;
    RandomMatrix<acc_type, 2> C;
    RandomMatrix<acc_type, 2> C_target;

    // Initialize matrices
    A.fill_rand<float_range>(m, k);
    B.fill_rand<float_range>(k, n);
    A_accT.fill_from(A, m, k);
    B_accT.fill_from(B, k, n);

    benchmark_kernel<acc_type, acc_type, 2, mm_kernel::register_tiled, false>(
        n_runs, m, n, k, A_accT, B_accT, C_target, C_target, std::string("GPU register tiled")
        );

    // TODO: make this work for Cutlass half, or cast?
//    benchmark_kernel<element_type, acc_type, 2, mm_kernel::tensor_naive, true>(
//        n_runs, m, n, k, A, B, C, C_target, std::string("GPU tensor naive")
//    );

    benchmark_kernel<element_type, acc_type, 2, mm_kernel::cublas, true>(
        n_runs, m, n, k, A, B, C, C_target, std::string("cublas")
        );

//    benchmark_kernel<element_type, acc_type, 2, mm_kernel::cutlass_default, true>(
//            n_runs, m, n, k, A, B, C, C_target, std::string("Cutlass default")
//    );

//    benchmark_kernel<element_type, acc_type, 2, mm_kernel::cutlass_custom, true>(
//            n_runs, m, n, k, A, B, C, C_target, std::string("Cutlass custom")
//    );

//    benchmark_kernel<element_type, acc_type, 2, mm_kernel::cutlass_simple, true>(
//            n_runs, m, n, k, A, B, C, C_target, std::string("Cutlass Simple")
//    );

    benchmark_kernel<element_type, acc_type, 2, mm_kernel::cute_mm, true>(
            n_runs, m, n, k, A, B, C, C_target, std::string("Cute")
    );

    benchmark_kernel<element_type, acc_type, 2, mm_kernel::tensor_optimized, true>(
            n_runs, m, n, k, A, B, C, C_target, std::string("GPU tensor optimized")
    );

    cudaFree(A.to_gpu());
    cudaFree(B.to_gpu());
    cudaFree(C.to_gpu());
    cudaFree(C_target.to_gpu());
    cudaFree(A_accT.to_gpu());
    cudaFree(B_accT.to_gpu());
#endif

    return 0;
}
