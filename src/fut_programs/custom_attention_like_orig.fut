-- ==
-- entry: run_origf16
-- compiled script input { (mk_inputf16 100000 16) }
-- compiled script input { (mk_inputf16 100000 32) }
-- compiled script input { (mk_inputf16 100000 64) }
-- compiled script input { (mk_inputf16 100000 128) }

-- ==
-- entry: run_origf32
-- compiled script input { (mk_inputf32 100000 16) }
-- compiled script input { (mk_inputf32 100000 32) }
-- compiled script input { (mk_inputf32 100000 64) }
-- compiled script input { (mk_inputf32 100000 128) }

import "mmm-helpers"

def oneIterf16 [d] (K: [d][d]f16) (V: [d][d]f16) (Qi: [d][d]f16) =
  let P_block = matmulf16 Qi K
  in matmulf16 P_block V

def flashAttentionf16 [m][d]
    (Q: [m][d][d]f16)
    (K: [d][d]f16)
    (V: [d][d]f16)  =
  map (oneIterf16 K V) Q

def oneIterf32 [d] (K: [d][d]f32) (V: [d][d]f32) (Qi: [d][d]f32) =
  let P_block = mmm_no_intra_f32 Qi K
  in mmm_no_intra_f32 P_block V

def flashAttentionf32 [m][d]
    (Q: [m][d][d]f32)
    (K: [d][d]f32)
    (V: [d][d]f32)  =
  map (oneIterf32 K V) Q

entry mk_inputf16 (m: i64) (d: i64) : ([m][d][d]f16, [d][d]f16, [d][d]f16) =
  let Q = replicate d 3.0 |> replicate d |> replicate m
  let K = replicate d 2.0 |> replicate d
  let V = replicate d 1.0 |> replicate d
  in  (Q, K, V)                             

entry run_origf16 [m][d] (Q: [m][d][d]f16) (K: [d][d]f16) (V: [d][d]f16) =
  flashAttentionf16 Q K V

entry mk_inputf32 (m: i64) (d: i64) : ([m][d][d]f32, [d][d]f32, [d][d]f32) =
  let Q = replicate d 3.0 |> replicate d |> replicate m
  let K = replicate d 2.0 |> replicate d
  let V = replicate d 1.0 |> replicate d
  in  (Q, K, V)                             

entry run_origf32 [m][d] (Q: [m][d][d]f32) (K: [d][d]f32) (V: [d][d]f32) =
  flashAttentionf32 Q K V



