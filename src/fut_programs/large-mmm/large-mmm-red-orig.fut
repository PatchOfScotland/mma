-- ==
-- entry: main
-- compiled random input {[8][16][128][64]f16 [16][8][64][128]f16}

-- compiled random input {[8][16][128][64]f16 [16][8][64][128]f16}
-- compiled random input {[16][32][128][64]f16 [32][16][64][128]f16}
-- compiled random input {[32][64][128][64]f16 [64][32][64][128]f16}
-- compiled random input {[64][128][128][64]f16 [128][64][64][128]f16} 



--                        [M] [K] [m]  [k]     [K] [N] [k] [n]
--                        [8] [16][128][64]    [16][8] [64][128]  
-- compiled random input {[32][32][128][64]f16 [32][32][64][128]f16}
-- compiled script input { (mk_input 32 32 32 128 128 64) }


import "mmm-intra-helpers"

let M = 8i64
let N = 8i64
let K = 16i64

let m = 128i64
let n = 128i64
let k = 64i64

entry mk_input M N K m n k : ([M][K][m][k]f16, [K][N][k][n]f16) =
  (replicate (M * K * m * k) 1f16 |> unflatten_4d, replicate (K * N * k * n) 1f16 |> unflatten_4d)

let ne () = (replicate (m * n) 0.0f32 |> unflatten)

-- TODO try different options, degrees of sequentialization, even just return elm
def reduceOp (acc: *[m][n]f32) (elm: [m][n]f32): [m][n]f32 =
  loop acc': *[m][n]f32 = (acc : *[m][n]f32) for i < m do
        acc' with [i, :] = map2 (+) elm[i] acc'[i]
--  ne ()

--[M][N][K]

def handleKBlocks (Arow: [K][m][k]f16) (Bcol: [K][k][n]f16) : [m][n]f32 =
    let acc_init = ne () in
--    map2 matmul Arow Bcol |>
--    reduce reduceOp acc_init
    loop (acc: *[m][n]f32) = acc_init for K_i < K do
        let C = matmul Arow[K_i] Bcol[K_i]
        in reduceOp acc C


entry main (A: [M][K][m][k]f16) (B: [K][N][k][n]f16) : [M][N][m][n]f32 =
-- TODO: extract to helper used in matmul?
    map (\Arow ->
        map (\Bcol ->
            handleKBlocks Arow Bcol
        ) (transpose B)
    ) A
