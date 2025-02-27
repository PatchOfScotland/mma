-- ==
-- entry: run_no_intra_f16
-- compiled random input {[1024][16][16]f16 [1024][256][16][16]f16}
-- compiled random input {[1024][32][32]f16 [1024][256][32][32]f16}
-- compiled random input {[1024][64][64]f16 [1024][256][64][64]f16}
-- only_intra compiled script input { (mk_input_f16 1024 256 128 128 128) }

-- == 
-- entry: run_no_intra_f32
-- compiled random input {[1024][16][16]f32 [1024][256][16][16]f32}
-- compiled random input {[1024][32][32]f32 [1024][256][32][32]f32}
-- compiled random input {[1024][64][64]f32 [1024][256][64][64]f32}
-- only_intra compiled script input { (mk_input_f32 1024 256 128 128 128) }


import "mmm-helpers"

def attention_like [q][m][n][k] 'a
  (mmm: [m][k]a -> [k][n]a -> [m][n]a)
  (ne : a)
  (A: [m][k]a)
  (B: [q][k][n]a) : [m][n]a =
  -- Copy to shared
  let A' = if q > 1
           then copy A
           else replicate (m * k) ne |> unflatten
  -- This is never worth it with cuda backend
  -- let A' = A
  let acc_init : *[m][n]a = replicate (m * n) ne |> unflatten in
  loop (_acc : *[m][n]a) = (acc_init: *[m][n]a) for i < q do
    let B': *[k][n]a = B[i]
    let C : *[m][n]a = mmm A' B'
    in copy C

       
def mk_input 'a (elm: a) (p: i64) (q: i64) (m: i64) (n: i64) (k: i64) =
  let A = replicate (p * m * k) elm |> unflatten_3d
  let B = replicate (p * q * k * n) elm |> unflatten_4d
  in (A, B)

entry mk_input_f16 = mk_input 1.0f16
entry mk_input_f32 = mk_input 1.0f32
                      
entry run_no_intra_f16 [q][p][d][k] (A: [p][d][k]f16) (B: [p][q][k][d]f16) =
  map2 (attention_like matmulf16 0.0f16) A B

entry run_no_intra_f32 [q][p][d][k] (A: [p][d][k]f32) (B: [p][q][k][d]f32) =
  map2 (attention_like mmm_no_intra_f32 0.0f32) A B
