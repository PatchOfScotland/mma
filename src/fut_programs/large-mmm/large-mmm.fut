-- ==
-- compiled random input {[32][32][128][64]f16 [32][32][64][128]f16}

-- compiled script input { (mk_input 32 32 32 128 128 64) }


import "mmm-intra-helpers"

let M = 32i64
let N = 32i64
let K = 32i64

let m = 128i64
let n = 128i64
let k = 64i64

entry mk_input M N K m n k : ([M][K][m][k]f16, [K][N][k][n]f16) =
  (replicate (M * K * m * k) 1f16 |> unflatten_4d, replicate (K * N * k * n) 1f16 |> unflatten_4d)

let ne () = (replicate (m * n) 0.0f32 |> unflatten)

-- TODO try different options, degrees of sequentialization, even just return elm
def reduceOp (acc: [m][n]f32) (elm: [m][n]f32): [m][n]f32 =
--  loop acc': *[m][n]f32 = (acc : *[m][n]f32) for i < m do
--        acc' with [i, :] = map2 (+) elm[i] acc'[i]
  ne ()

--[M][N][K]

def handleKBlocks (Arow: [K][m][k]f16) (Bcol: [K][k][n]f16) : [m][n]f32 =
--    map2 matmul Arow Bcol |>
--    #[incremental_flattening(only_intra)]map2 matmul Arow Bcol |>
--    reduce reduceOp (ne ())
--    #[incremental_flattening(only_intra)]map2 matmul Arow Bcol
--    map2 matmul Arow Bcol
    let acc_init = ne () in
    loop (acc: [m][n]f32) = acc_init for K_i < K do
      matmul Arow[K_i] Bcol[K_i]

entry main (A: [M][K][m][k]f16) (B: [K][N][k][n]f16) : [M][N][m][n]f32 =
-- TODO: extract to helper used in matmul?
    #[incremental_flattening(only_inner)]map (\Arow ->
        #[incremental_flattening(only_intra)]map (\Bcol ->
            handleKBlocks Arow Bcol
        ) (transpose B)
    ) A
