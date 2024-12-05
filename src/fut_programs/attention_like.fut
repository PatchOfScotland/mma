-- ==
-- compiled script input { (mk_input 10000 128) }
import "mmm-helpers"

let d = 128i64

-- Note: Due to a compiler bug (described in the compiler)
-- this will give the wrong results
def oneIter (K: [d][d]f16) (V: [d][d]f16) (Qi: [d][d]f16) =
  let P_block = matmulf16 Qi K
  in matmulf16 P_block V

def flashAttention [m]
    (Q: [m][d][d]f16)
    (K: [d][d]f16)
    (V: [d][d]f16)  =
  map (oneIter K V) Q

entry mk_input (m: i64) (d: i64) : ([m][d][d]f16, [d][d]f16, [d][d]f16) =
  let Q = replicate d 3.0 |> replicate d |> replicate m
  let K = replicate d 2.0 |> replicate d
  let V = replicate d 1.0 |> replicate d
  in  (Q, K, V)


entry main [m] (Q: [m][d][d]f16) (K: [d][d]f16) (V: [d][d]f16) =
  -- let (Q, K, V) = mk_input 1 d in
  #[incremental_flattening(only_intra)]flashAttention Q K V
