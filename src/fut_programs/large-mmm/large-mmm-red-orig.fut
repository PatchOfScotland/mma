-- ==
-- entry: main
-- "[8][16][128][64]f16 [16][8][64][128]f16" script input { (mk_input 8i64 8i64 16i64 128i64 64i64 128i64) }
-- "[16][32][128][64]f16 [32][16][64][128]f16" script input { (mk_input 16i64 16i64 32i64 128i64 64i64 128i64) }
-- "[32][64][128][64]f16 [64][32][64][128]f16" script input { (mk_input 32i64 32i64 64i64 128i64 64i64 128i64) }
-- "[64][128][128][64]f16 [128][64][64][128]f16" script input { (mk_input 64i64 64i64 128i64 128i64 64i64 128i64) }


entry mk_input (M:i64) (N:i64) (K:i64) (m:i64) (n:i64) (k:i64) : ([M][K][m][k]f16, [K][N][k][n]f16) =
  (replicate (M * K * m * k) 1f16 |> unflatten_4d, replicate (K * N * k * n) 1f16 |> unflatten_4d)


import "mmm-intra-helpers"

-- TODO try different options, degrees of sequentialization, even just return elm
def matAdd [m][n] (acc: *[m][n]f32) (elm: [m][n]f32): [m][n]f32 =
  loop acc': *[m][n]f32 = (acc : *[m][n]f32) for i < m do
        acc' with [i, :] = map2 (+) elm[i] acc'[i]
--  ne ()

--[M][N][K]

def handleKBlocks [K][m][n][k] (Arow: [K][m][k]f16) (Bcol: [K][k][n]f16) : [m][n]f32 =
    let acc_init = (replicate (m * n) 0.0f32 |> unflatten) in
--    map2 matmul Arow Bcol |>
--    reduce matAdd acc_init
    loop (acc: *[m][n]f32) = acc_init for K_i < K do
        let C = matmul Arow[K_i] Bcol[K_i]
        in matAdd acc C


entry main [M][N][K][m][n][k] (A: [M][K][m][k]f16) (B: [K][N][k][n]f16) : [M][N][m][n]f32 =
-- TODO: extract to helper used in matmul?
    map (\Arow ->
        map (\Bcol ->
            handleKBlocks Arow Bcol
        ) (transpose B)
    ) A
