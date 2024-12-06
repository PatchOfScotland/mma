-- ==
-- entry: run_orig
-- compiled script input { (mk_input 100000 16) }
-- compiled script input { (mk_input 100000 32) }
-- compiled script input { (mk_input 100000 64) }
-- compiled script input { (mk_input 100000 128) }

import "custom_attention_like"

entry mk_input (m: i64) (d: i64) : ([m][d][d]real, [d][d]real, [d][d]real) =
  let Q = replicate d 3.0 |> replicate d |> replicate m
  let K = replicate d 2.0 |> replicate d
  let V = replicate d 1.0 |> replicate d
  in  (Q, K, V)                             

entry run_orig [m][d] (Q: [m][d][d]real) (K: [d][d]real) (V: [d][d]real) =
  flashAttention Q K V
