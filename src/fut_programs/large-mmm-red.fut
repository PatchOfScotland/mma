-- ==
-- compiled random input {[16][16][128][64]f16 [16][16][64][128]f16}
-- compiled random input {[32][32][128][64]f16 [32][32][64][128]f16}
-- compiled random input {[32][64][128][64]f16 [64][32][64][128]f16}

-- compiled script input { (mk_input 32 32 32 128 128 64) }


import "mmm-intra-helpers"

-- let M = 32i64
-- let N = 32i64
-- let K = 32i64

-- let m = 128i64
-- let n = 128i64
-- let k = 64i64

entry mk_input M N K m n k : ([M][K][m][k]f16, [K][N][k][n]f16) =
  (replicate (M * K * m * k) 1f16 |> unflatten_4d, replicate (K * N * k * n) 1f16 |> unflatten_4d)

let ne (m: i64) (n: i64) = (replicate (m * n) 0.0f32 |> unflatten)

-- TODO try different options, degrees of sequentialization, even just return elm
def reduceOp [m][n] (acc: *[m][n]f32) (elm: [m][n]f32): [m][n]f32 =
  loop acc': *[m][n]f32 = (acc : *[m][n]f32) for i < m do
        acc' with [i, :] = map2 (+) elm[i] acc'[i]
--  ne ()

--[M][N][K]

def handleKBlocks[K][m][n][k] (Arow: [K][m][k]f16) (Bcol: [K][k][n]f16) : [m][n]f32 =
    let acc_init = ne m n in
--    map2 matmul Arow Bcol |>
--    reduce reduceOp acc_init
    loop (acc: *[m][n]f32) = acc_init for K_i < K do
        let C = matmul Arow[K_i] Bcol[K_i]
        in reduceOp acc C


def run [M][K][N][m][n][k] (A: [M][K][m][k]f16) (B: [K][N][k][n]f16) : [M][N][m][n]f32 =
-- TODO: extract to helper used in matmul?
    #[incremental_flattening(only_inner)]map (\Arow ->
        #[incremental_flattening(only_intra)]map (\Bcol ->
            handleKBlocks Arow Bcol
        ) (transpose B)
    ) A

entry run_small (A: [16][16][128][64]f16) (B: [16][16][64][128]f16) = run A B 
entry run_medium (A: [32][32][128][64]f16) (B: [32][32][64][128]f16) = run A B
entry run_large (A: [32][64][128][64]f16) (B: [64][32][64][128]f16) = run A B 
        

