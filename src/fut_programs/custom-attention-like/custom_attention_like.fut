-- ==
-- entry: run16
-- only_intra compiled script input { (mk_input 100000 16) }

-- ==
-- entry: run32
-- only_intra compiled script input { (mk_input 100000 32) }

-- ==
-- entry: run64
-- only_intra compiled script input { (mk_input 100000 64) }

-- ==
-- entry: run128
-- only_intra compiled script input { (mk_input 100000 128) }

import "mmm-helpers"

type real = f16

def matmul [d] (A: [d][d]real) (B: [d][d]real) : [d][d]real =
  map (\Arow ->
         map (\Bcol ->
                map2 (*) Arow Bcol
                |> reduce (+) 0.0
             ) (transpose B)
      ) A

-- Note: Due to a compiler bug (described in the compiler)
-- this will give the wrong results
def oneIter [d] (K: [d][d]real) (V: [d][d]real) (Qi: [d][d]real) =
  let P_block = matmul Qi K
  in matmul P_block V

def flashAttention [m][d]
    (Q: [m][d][d]real)
    (K: [d][d]real)
    (V: [d][d]real)  =
  map (oneIter K V) Q

entry mk_input (m: i64) (d: i64) : ([m][d][d]real, [d][d]real, [d][d]real) =
  let Q = replicate d 3.0 |> replicate d |> replicate m
  let K = replicate d 2.0 |> replicate d
  let V = replicate d 1.0 |> replicate d
  in  (Q, K, V)

entry run16 [m] (Q: [m][16][16]real) (K: [16][16]real) (V: [16][16]real) =
  #[incremental_flattening(only_intra)]flashAttention Q K V

entry run32 [m] (Q: [m][32][32]real) (K: [32][32]real) (V: [32][32]real) =
  #[incremental_flattening(only_intra)]flashAttention Q K V

entry run64 [m] (Q: [m][64][64]real) (K: [64][64]real) (V: [64][64]real) =
  #[incremental_flattening(only_intra)]flashAttention Q K V

entry run128 [m] (Q: [m][128][128]real) (K: [128][128]real) (V: [128][128]real) =
  #[incremental_flattening(only_intra)]flashAttention Q K V

