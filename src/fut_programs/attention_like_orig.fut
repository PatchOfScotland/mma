-- ==
-- entry: run_no_intra
-- compiled random input {[1024][16][16]f16 [1024][256][16][16]f16}
-- compiled random input {[1024][32][32]f16 [1024][256][32][32]f16}
-- compiled random input {[1024][64][64]f16 [1024][256][64][64]f16}
-- only_intra compiled script input { (mk_input 1024 256 128 128 64) }

import "mmm-helpers"
type real = f16

def attention_like [q][m][n][k] (A: [m][k]f16) (B: [q][k][n]f16) : [m][n]f16 =
  -- Copy to shared
  let A' = if q > 1
           then copy A
           else replicate (m * k) 0.0f16 |> unflatten

  let acc_init : *[m][n]f16 = replicate (m * n) 0.0f16 |> unflatten in
  loop (_acc : *[m][n]f16) = (acc_init: *[m][n]f16) for i < q do
    let B': *[k][n]f16 = B[i]
    let C : *[m][n]f16 = matmulf16 A' B'
    in copy C

       
entry mk_input (p: i64) (q: i64) (m: i64) (n: i64) (k: i64) =
  let A = replicate (p * m * k) 1.0f16 |> unflatten_3d
  let B = replicate (p * q * k * n) 1.0f16 |> unflatten_4d
  in (A, B)
                      
entry run_no_intra [q][p][d] (A: [p][d][d]f16) (B: [p][q][d][d]f16) =
  map2 attention_like A B
