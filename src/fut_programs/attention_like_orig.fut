-- ==
-- entry: run_no_intra
-- compiled random input {[1024][16][16]f16 [1024][256][16][16]f16}
-- compiled random input {[1024][32][32]f16 [1024][256][32][32]f16}
-- compiled random input {[1024][64][64]f16 [1024][256][64][64]f16}
-- only_intra compiled script input { (mk_input 1024 256 128 128 128) }

import "attention_like"
       
entry mk_input (p: i64) (q: i64) (m: i64) (n: i64) (k: i64) =
  let A = replicate (p * m * k) 1.0f16 |> unflatten_3d
  let B = replicate (p * q * k * n) 1.0f16 |> unflatten_4d
  in (A, B)
                      
entry run_no_intra [q][p][d] (A: [p][d][d]f16) (B: [p][q][d][d]f16) =
  map2 attention_like A B
