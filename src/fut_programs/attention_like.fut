-- ==
-- compiled script input { (mk_input 1 16) } auto output

let d = 16i64

def matmulf32 (A: [d][d]f16) (B: [d][d]f16) : [d][d]f32 =
  map (\Arow ->
         map (\Bcol ->
                map2 (*) Arow Bcol
                |> map f32.f16
                |> reduce (+) 0.0
             ) (transpose B)
      ) A

def matmulf16 (A: [d][d]f16) (B: [d][d]f16) : [d][d]f16 =
  map (\Arow ->
         map (\Bcol ->
                map2 (*) Arow Bcol
                |> reduce (+) 0.0
             ) (transpose B)
      ) A

def oneIter (K: [d][d]f16) (V: [d][d]f16) (Qi: [d][d]f16) =
  let P_block = matmulf16 Qi K
  -- in P_block
  in matmulf16 P_block V

def flashAttention [m]
    (Q: [m][d][d]f16)
    (K: [d][d]f16)
    (V: [d][d]f16)  =
  map (oneIter K V) Q

entry mk_input (m: i64) (d: i64) : ([m][d][d]f16, [d][d]f16, [d][d]f16) =
  let Q = replicate d 1.0 |> replicate d |> replicate m
  let K = replicate d 1.0 |> replicate d
  let V = replicate d 1.0 |> replicate d
  in  (Q, K, V)


entry main [m] (Q: [m][d][d]f16) (K: [d][d]f16) (V: [d][d]f16) =
  #[incremental_flattening(only_intra)]flashAttention Q K V
