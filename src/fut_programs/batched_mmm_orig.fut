-- ==
-- entry: mmm
-- compiled script input { (mk_input 100000 16 16) } auto output
-- compiled random input { (mk_input 100000 32 32) } auto output
-- compiled random input { (mk_input 100000 64 64) } auto output
-- compiled random input { (mk_input 100000 128 128) } auto output

import "batched_mmm"
import "mmm-helpers"

entry mk_input (q: i64) (m: i64) (n: i64) (k: i64) =
  let A = replicate (q * m * k) 1.0f16 |> unflatten_3d
  let B = replicate (q * q * k * n) 1.0f16 |> unflatten_3d
  in (A, B)                   

entry mmm [q] [d] (A: [q][d][d]f16) (B: [q][d][d]f16) : [q][d][d]f32 =
  map2 matmulf32 A B
                   
