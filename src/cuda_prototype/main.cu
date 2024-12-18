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

#include "cutlass/half.h"
#include <cublas_v2.h>


// Most useful macros:

// For setting configuration values:
// FRAGS_M, FRAGS_N, FRAGS_K
// BLOCK_TILES_M, BLOCK_TILES_N
// NUM_STAGES
// Only pure CUDA:
// WARP_TILES_M, WARP_TILES_N, WARP_TILES_K, SHARED_PADDING

// For enabling/disabling optimizations:
// SYNC_CPY, NO_SWIZZLE
// Only CuTe:
// NO_LDSM, NO_VECTORIZE,


#define WARP_SIZE 32
#define SHARED_MEM_SIZE 49152
#define MAX_THREADS_PER_BLOCK 1024
#define MAX_REGISTERS_PER_BLOCK 65536

#ifndef SHARED_PADDING
#define SHARED_PADDING 8
#endif

#ifndef NUM_STAGES
#define NUM_STAGES 3
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
#define BLOCK_TILES_N 4
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

    #ifdef NO_SWIZZLE
    constexpr unsigned int shared_memory_used_A = shared_m * (shared_k + SHARED_PADDING) * sizeof(elmT) * NUM_STAGES;
    constexpr unsigned int shared_memory_used_B = shared_k * (shared_n + SHARED_PADDING) * sizeof(elmT) * NUM_STAGES;
    #else
    constexpr unsigned int shared_memory_used_A = shared_m * shared_k * sizeof(elmT) * NUM_STAGES;
    constexpr unsigned int shared_memory_used_B = shared_k * shared_n * sizeof(elmT) * NUM_STAGES;
    #endif

    constexpr unsigned int shared_memory_used = shared_memory_used_A + shared_memory_used_B;

    printf("    Shared memory used: %d/%d bytes (%.0f%%)\n", shared_memory_used, max_shared_memory, (float) shared_memory_used / max_shared_memory * 100);
    printf("    Shared memory used A: %d/%d bytes (%.0f%%)\n", shared_memory_used_A, max_shared_memory, (float) shared_memory_used_A / max_shared_memory * 100);
    printf("    Shared memory used B: %d/%d bytes (%.0f%%)\n", shared_memory_used_B, max_shared_memory, (float) shared_memory_used_B / max_shared_memory * 100);

    auto kernel = matMulTiledTensor<elmT, elmAccT, WMMA_M, WMMA_N, WMMA_K, FRAGS_M, FRAGS_N, FRAGS_K, WARP_TILES_M, WARP_TILES_N, WARP_TILES_K, BLOCK_TILES_M, BLOCK_TILES_N, threads_per_block, NUM_STAGES>;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_used);

    TimeMeasurement t;

    t.start();
    for (int i = 0; i < n_runs; i++) {
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


template <typename elmT, typename elmAccT = elmT>
long int benchmark_cute_mmm(int n_runs, elmT * A, elmT * B, elmAccT * C, int m, int n, int k);

template<>
long int benchmark_cute_mmm<half_t, float>(int n_runs, half_t * A, half_t * B, float * C, int m, int n, int k) {
    using namespace cute;

    using TA = half_t;
    using TB = half_t;
    using TC = float;

    // mma tiling
    TiledMMA tiled_mma = make_tiled_mma(
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>{},
        Layout<Shape<Int<BLOCK_TILES_M>,Int<BLOCK_TILES_N>,_1>>{},
        Tile<Int<BLOCK_TILES_M * WMMA_M>, Int<BLOCK_TILES_N * WMMA_N>, Int<WMMA_K>>{}
    );

    // Full matrix shape and stride
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto dA = make_stride(K, Int<1>{});
    auto dB = make_stride(Int<1>{}, N);
    auto dC = make_stride(N, Int<1>{});
    auto prob_shape = make_shape(M, N, K);

    // Shared memory layout
    auto bM = Int<WMMA_M * FRAGS_M * WARP_TILES_M * BLOCK_TILES_M>{};
    auto bN = Int<WMMA_N * FRAGS_N * WARP_TILES_N * BLOCK_TILES_N>{};
    auto bK = Int<WMMA_K * FRAGS_K * WARP_TILES_K>{};
    auto bP = Int<NUM_STAGES>{};
    auto cta_tiler = make_shape(bM, bN, bK);

    using SharedM = decltype(bM);
    using SharedN = decltype(bN);
    using SharedK = decltype(bK);

    using layoutAtom_A = Layout<
        Shape<SharedM, SharedK>,
        Stride<SharedK, _1>
    >;
    using layoutAtom_B = Layout<
        Shape<SharedN, SharedK>,
        Stride<_1, SharedN>
    >;

    constexpr unsigned int sizeKunsigned = bK;
    constexpr unsigned int shift_lenK = max(bit_width(sizeKunsigned) - 4, _3{});
    constexpr unsigned int sizeNunsigned = bN;
    constexpr unsigned int shift_lenN = max(bit_width(sizeNunsigned) - 4, _3{});

#ifdef NO_SWIZZLE
    auto swizzle_layoutAtom_A = layoutAtom_A{};
    auto swizzle_layoutAtom_B = layoutAtom_B{};
#else
    auto swizzle_layoutAtom_A = composition(Swizzle<3,3,shift_lenK>{}, layoutAtom_A{});
    auto swizzle_layoutAtom_B = composition(Swizzle<3,3,shift_lenN>{}, layoutAtom_B{});
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

    // global->shared copy tiling
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

    TiledCopy copyA_global_shared = make_tiled_copy(Copy_Atom<ACopyOpGlobalShared, TA>{},
        Layout<
            Shape<Int<BLOCK_TILES_M * BLOCK_TILES_N * WARP_SIZE / (WMMA_K * FRAGS_K * WARP_TILES_K / elms_per_load)>, Int<WMMA_K * FRAGS_K * WARP_TILES_K / elms_per_load>>,
            Stride<Int<WMMA_K * FRAGS_K * WARP_TILES_K / elms_per_load>, _1>
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

    auto sC = composition(Swizzle<3,2,shift_lenN>{}, make_layout(make_shape(bM, bN), LayoutRight{}));

    // shared->register copy tiling
#ifdef NO_LDSM
    using ACopyOpSharedRegisters = AutoVectorizingCopy;
    using BCopyOpSharedRegisters = AutoVectorizingCopy;
#else
    using ACopyOpSharedRegisters = SM75_U32x4_LDSM_N;
    using BCopyOpSharedRegisters = SM75_U16x8_LDSM_T;
#endif

    TiledCopy copyA_shared_registers = make_tiled_copy_A(Copy_Atom<ACopyOpSharedRegisters, TA>{}, tiled_mma);
    TiledCopy copyB_shared_registers = make_tiled_copy_B(Copy_Atom<BCopyOpSharedRegisters, TB>{}, tiled_mma);

#if (NUM_STAGES == 1)
    auto kernel = gemm_simple<
            decltype(prob_shape), decltype(cta_tiler),
            TA, decltype(dA), decltype(sA), decltype(copyA_global_shared), decltype(copyA_shared_registers),
            TB, decltype(dB), decltype(sB), decltype(copyB_global_shared), decltype(copyB_shared_registers),
            TC, decltype(dC), decltype(sC), decltype(tiled_mma)
    >;
#else
#ifdef SYNC_CPY
    static_assert(NUM_STAGES == 2, "NUM_STAGES must be 2 for gemm_sync_cpy");

    auto kernel = gemm_sync_cpy<
            decltype(prob_shape), decltype(cta_tiler),
            TA, decltype(dA), decltype(sA), decltype(copyA_global_shared), decltype(copyA_shared_registers),
            TB, decltype(dB), decltype(sB), decltype(copyB_global_shared), decltype(copyB_shared_registers),
            TC, decltype(dC), decltype(sC), decltype(tiled_mma)
    >;
#else
    auto kernel = gemm_pipelined<
            decltype(prob_shape), decltype(cta_tiler),
            TA, decltype(dA), decltype(sA), decltype(copyA_global_shared), decltype(copyA_shared_registers),
            TB, decltype(dB), decltype(sB), decltype(copyB_global_shared), decltype(copyB_shared_registers),
            TC, decltype(dC), decltype(sC), decltype(tiled_mma),
            NUM_STAGES
    >;
#endif
#endif

#ifdef SWIZZLE_BACK
#ifdef SEPARATE_CMEM
    const uint32_t shared_memory_used = cosize_v<decltype(sA)> * sizeof(TA) + cosize_v<decltype(sB)> * sizeof(TB) + cosize_v<decltype(sC)> * sizeof(TC);
#else
    const uint32_t shared_memory_used = max(cosize_v<decltype(sA)> * sizeof(TA) + cosize_v<decltype(sB)> * sizeof(TB), cosize_v<decltype(sC)> * sizeof(TC));
#endif
#else
    const uint32_t shared_memory_used = cosize_v<decltype(sA)> * sizeof(TA) + cosize_v<decltype(sB)> * sizeof(TB);
#endif

    // Set dynamic shared memory size
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_used);
    dim3 dimBlock(size(tiled_mma));
    dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));

    printf("Used shared memory: %d, threads: %d\n", shared_memory_used, (int)(size(tiled_mma)));

    TimeMeasurement t;
    t.start();
    for (int i = 0; i < n_runs; i++) {
        kernel<<<dimGrid, dimBlock, shared_memory_used>>>(
                prob_shape,
                A, dA,
                B, dB,
                C, dC
        );
    }
    cudaDeviceSynchronize();
    t.stop();

    gpuAssert(cudaPeekAtLastError());
    return t.elapsed();
}

template <typename elmT, typename elmAccT = elmT>
long int benchmark_cute_attention_like(unsigned int n_runs, elmT * As, elmT * Bss, elmAccT * Cs, unsigned int batches, unsigned int reuse);

template<>
long int benchmark_cute_attention_like<half_t, float>(unsigned int n_runs, half_t * As, half_t * Bss, float * Cs, unsigned int batches, unsigned int reuse) {
    using namespace cute;

    using TA = half_t;
    using TB = half_t;
    using TC = float;

    // mma tiling
    TiledMMA tiled_mma = make_tiled_mma(
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>{},
        Layout<Shape<Int<BLOCK_TILES_M>,Int<BLOCK_TILES_N>,_1>>{},
        Tile<Int<BLOCK_TILES_M * WMMA_M>, Int<BLOCK_TILES_N * WMMA_N>, Int<WMMA_K>>{}
    );

    // Shared memory layout
    auto bM = Int<WMMA_M * FRAGS_M * WARP_TILES_M * BLOCK_TILES_M>{};
    auto bN = Int<WMMA_N * FRAGS_N * WARP_TILES_N * BLOCK_TILES_N>{};
    auto bK = Int<WMMA_K * FRAGS_K * WARP_TILES_K>{};
    auto bP = Int<NUM_STAGES>{};

    using SharedM = decltype(bM);
    using SharedN = decltype(bN);
    using SharedK = decltype(bK);

    auto layoutAs = make_layout(make_shape(bM, bK, batches), make_stride(bK, Int<1>{}, bM * bK));
#ifdef BATCHED
    auto layoutBss = make_layout(make_shape(bN, bK, batches, Int<1>{}), make_stride(Int<1>{}, bN, bN * bK, Int<0>{}));
#else
    auto layoutBss = make_layout(make_shape(bN, bK, batches, reuse), make_stride(Int<1>{}, bN, bN * bK * reuse, bN * bK));
#endif
    auto layoutCs = make_layout(make_shape(bM, bN, batches), make_stride(bN, Int<1>{}, bM * bN));

    using layoutAtom_A = Layout<
        Shape<SharedM, SharedK>,
        Stride<SharedK, _1>
    >;
    using layoutAtom_B = Layout<
        Shape<SharedN, SharedK>,
        Stride<_1, SharedN>
    >;

    constexpr unsigned int sizeKunsigned = bK;
    constexpr unsigned int shift_lenK = max(bit_width(sizeKunsigned) - 4, _3{});
    constexpr unsigned int sizeNunsigned = bN;
    constexpr unsigned int shift_lenN = max(bit_width(sizeNunsigned) - 4, _3{});

#ifdef NO_SWIZZLE
    auto swizzle_layoutAtom_A = layoutAtom_A{};
    auto swizzle_layoutAtom_B = layoutAtom_B{};
#else
    auto swizzle_layoutAtom_A = composition(Swizzle<3,3,shift_lenK>{}, layoutAtom_A{});
    auto swizzle_layoutAtom_B = composition(Swizzle<3,3,shift_lenN>{}, layoutAtom_B{});
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

    // global->shared copy tiling
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

    auto sC = composition(Swizzle<3,2,shift_lenN>{}, make_layout(make_shape(bM, bN), LayoutRight{}));

    // global->shared copy tiling
    TiledCopy copyA_global_shared = make_tiled_copy(Copy_Atom<ACopyOpGlobalShared, TA>{},
        Layout<
            Shape<Int<BLOCK_TILES_M * BLOCK_TILES_N * WARP_SIZE / (WMMA_K * FRAGS_K * WARP_TILES_K / elms_per_load)>, Int<WMMA_K * FRAGS_K * WARP_TILES_K / elms_per_load>>,
            Stride<Int<WMMA_K * FRAGS_K * WARP_TILES_K / elms_per_load>, _1>
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
    #ifdef ATTENTION_LIKE
    static_assert(NUM_STAGES == 1, "NUM_STAGES must be 1 for attention_like");

    auto kernel = attention_like_simple<
        TA, decltype(layoutAs), decltype(sA), decltype(copyA_global_shared), decltype(copyA_shared_registers),
        TB, decltype(layoutBss), decltype(sB), decltype(copyB_global_shared), decltype(copyB_shared_registers),
        TC, decltype(layoutCs), decltype(sC), decltype(tiled_mma)
    >;

#ifdef SWIZZLE_BACK
#ifdef SEPARATE_CMEM
    const uint32_t shared_memory_used = cosize_v<decltype(sA)> * sizeof(TA) + cosize_v<decltype(sB)> * sizeof(TB) + cosize_v<decltype(sC)> * sizeof(TC);
#else
    const uint32_t shared_memory_used = max(cosize_v<decltype(sA)> * sizeof(TA) + cosize_v<decltype(sB)> * sizeof(TB), cosize_v<decltype(sC)> * sizeof(TC));
#endif
#else
    const uint32_t shared_memory_used = cosize_v<decltype(sA)> * sizeof(TA) + cosize_v<decltype(sB)> * sizeof(TB);
#endif

    // Set dynamic shared memory size
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_used);
    dim3 dimBlock(size(tiled_mma));
    dim3 dimGrid(batches);

    printf("Used shared memory: %d, threads: %d\n", shared_memory_used, (int)(size(tiled_mma)));

    TimeMeasurement t;
    t.start();
    for (int i = 0; i < n_runs; i++) {
        kernel<<<dimGrid, dimBlock, shared_memory_used>>>(
            As, layoutAs,
            Bss, layoutBss,
            Cs, layoutCs
        );
    }
    cudaDeviceSynchronize();
    t.stop();

    gpuAssert(cudaPeekAtLastError());
    return t.elapsed();
    #else
    return 0;
    #endif
}

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

    int dimx = ceil(((float) n)/(wmma_n * block_tiles_n));
    int dimy = ceil( ((float) m)/(wmma_m * block_tiles_m));
    dim3 grid(dimx, dimy, 1);

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

    benchmark_kernel<element_type, acc_type, 2, mm_kernel::cublas, true>(
        n_runs, m, n, k, A, B, C, C_target, std::string("cublas")
    );

    benchmark_kernel<element_type, acc_type, 2, mm_kernel::tensor_optimized, true>(
            n_runs, m, n, k, A, B, C, C_target, std::string("GPU tensor optimized")
    );

    benchmark_kernel<element_type, acc_type, 2, mm_kernel::cute_mm, true>(
            n_runs, m, n, k, A, B, C, C_target, std::string("Cute")
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
