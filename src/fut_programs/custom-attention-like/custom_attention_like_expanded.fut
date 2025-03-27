-- ==
-- entry: run16
-- only_intra compiled script input { (mk_input 256 16) }

-- ==
-- entry: run32
-- only_intra compiled script input { (mk_input 256 32) }

-- ==
-- entry: run64
-- only_intra compiled script input { (mk_input 256 64) }

-- ==
-- entry: run128
-- only_intra compiled script input { (mk_input 256 128) }

import "mmm-helpers"

type real = f16

def matmul [m][n][k] (A: [m][k]real) (B: [k][n]real) : [m][n]real =
  map (\Arow ->
    map (\Bcol ->
      map2 (*) Arow Bcol
        |> reduce (+) 0.0
    ) (transpose B)
  ) A

-- Note: Due to a compiler bug (described in the compiler)
-- this will give the wrong results
def oneIter [m][d] (K: [m*d][d]real) (V: [m*d][d]real) (Qi: [d][d]real) =
  let P_block = matmul Qi (transpose K)
  in matmul P_block V

def flashAttention [m][d]
    (Q: [m][d][d]real)
    (K: [m*d][d]real)
    (V: [m*d][d]real)  =
  map (oneIter K V) Q

entry mk_input (m: i64) (d: i64) : ([m][d][d]real, [m*d][d]real, [m*d][d]real) =
  let Q = replicate d 3.0 |> replicate d |> replicate m
  let K = replicate d 2.0 |> replicate (m*d)
  let V = replicate d 1.0 |> replicate (m*d)
  in  (Q, K, V)

entry run16 [m] (Q: [m][16][16]real) (K: [m*16][16]real) (V: [m*16][16]real) =
  flashAttention Q K V

entry run32 [m] (Q: [m][32][32]real) (K: [m*32][32]real) (V: [m*32][32]real) =
  flashAttention Q K V

entry run64 [m] (Q: [m][64][64]real) (K: [m*64][64]real) (V: [m*64][64]real) =
  flashAttention Q K V

entry run128 [m] (Q: [m][128][128]real) (K: [m*128][128]real) (V: [m*128][128]real) =
  flashAttention Q K V
