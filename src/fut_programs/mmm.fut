-- ==
-- entry: validate
-- nobench compiled input { 11i64 } output { true }
-- nobench compiled input { 42i64 } output { true }
-- nobench compiled input { 256i64 } output { true }

-- entry: matmul_bench
-- compiled script intput { (mk_input 256) } 

import "mmm-helpers"     

let d = 16i64

def matmul [k] (A : [k * d][k * d]f16) (B : [k * d][k * d]f16) =
  #[incremental_flattening(only_intra)]
  tabulate k (\i ->
                tabulate k (\j ->
                              loop c = replicate d (replicate d 0.0f16)
                              for k' < k do
                              let A' = copy A[i:i+d, k':k'+d] :> [d][d]f16
                              let B' = copy B[k':k'+d, j:j+d] :> [d][d]f16
                              let C = matmulf16 A' B'
                              in map2 (map2 (+)) c C
                           )
             )

def reshape [k] (A : [k * d][k * d]f16) : [k * k][d][d]f16 =
  flatten A |> sized (k * k * d * d) |> unflatten_3d
  
def matmul2 [k] (A : [k * d][k * d]f16) (B : [k * d][k * d]f16) =
  #[incremental_flattening(only_intra)]
  tabulate k (\i ->
                tabulate k (\j ->
                              let i' = i * d
                              let j' = j * d in
                              loop acc = replicate d (replicate d 0.0f16)
                              for q < k do
                              let k' = q * d
                              let A' = A[i':i'+d, k':k'+d] :> [d][d]f16
                              let B' = B[k':k'+d, j':j'+d] :> [d][d]f16
                              let C = matmulf16 A' B'
                              in map2 (map2 (*)) acc C
                           )                
             )
-- def mmm [k] (A : [k * d][k * d]f16) (B : [k * d][k * d]f16) =
--   map (\Arow ->
--          map (\Bcol ->
--                 map2 (*) Arow Bcol
--                 |> reduce (+) 0
--              ) (transpose B)
--       ) A

-- entry mk_input (k: i64) =
--   let A = replicate (k * d) (replicate (k * d) 1.0f16)
--   let B = replicate (k * d) (replicate (k * d) 1.0f16)
--   in (A, B)

-- entry validate (k: i64) =
--   let (A, B) = mk_input k
--   let C_actual = matmul A B |> flatten |> flatten |> flatten
--   let C_expected = mmm A B |> flatten :> [k * k * d * d]f16 in
--   map2 (==) C_actual C_expected
--   |> and

entry matmul_bench (A : [256i64 * d][256i64 * d]f16) (B : [256i64 * d][256i64 * d]f16) = matmul2 A B  
    
-- def main () =
--   let A = replicate (2 * d) (replicate (2*d) 1.0f16)
--   let B = replicate (2 * d) (replicate (2*d) 1.0f16)
--   in matmul A B
      
