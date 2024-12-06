-- ==
-- entry: run_no_intra
-- compiled random input {[1024][16][16]f16 [1024][256][16][16]f16}
-- compiled random input {[1024][32][32]f16 [1024][256][32][32]f16}
-- compiled random input {[1024][64][64]f16 [1024][256][64][64]f16}
-- compiled random input {[1024][128][128]f16 [1024][256][128][128]f16}

-- ==
-- entry: run16
-- only_intra compiled random input {[1024][16][16]f16 [1024][256][16][16]f16}

-- ==
-- entry: run32
-- only_intra compiled random input {[1024][32][32]f16 [1024][256][32][32]f16}

-- ==
-- entry: run64
-- only_intra compiled random input {[1024][64][64]f16 [1024][256][64][64]f16}

-- ==
-- entry: run128
-- only_intra compiled script input { (mk_input 1024 256 128 128 128) }
import "mmm-helpers"


def seq_acc [m][n] (acc: *[m][n]f32) (C: *[m][n]f32) =
  let acc_flat = flatten acc
  let Cflat = flatten C
  let thrd_work = m * n / 32 in
  let res = tabulate 32 (\thrd_idx ->
                           let start = thrd_idx * thrd_work in
                           loop acc' = replicate thrd_work 0.0f32
                           for i < thrd_work do
                             acc' with [i] = acc_flat[start + i] + Cflat[start + i]
                        )
  let res_flat = sized (m * n) (flatten res)
  in unflatten res_flat

def seq_acc2 [m][n] (acc: *[m][n]f32) (C: *[m][n]f32) =
  let thrd_work = m * n / 32 in
  loop acc' = acc for j < thrd_work do
  let js_per_row = n / 32 in
  let row = j / js_per_row
  let col = j % js_per_row
  let col_offset = col * 32
  in acc' with [row, col_offset:col_offset + 32] =
    tabulate 32 (\k -> acc'[row, col_offset + k] + C[row, col_offset + k])

let seq_acc3 [m][n] (acc: *[m][n]f32) (C: *[m][n]f32) =
  loop acc': *[m][n]f32 = (acc : *[m][n]f32) for i < m do
      acc' with [i, :] = map2 (+) C[i] acc'[i]

let seq_acc4 [m][n] (acc: *[m][n]f32) (C: *[m][n]f32) =
  loop acc': *[m][n]f32 = (acc : *[m][n]f32) for i < m do
    acc' with [i, :] = map2 (+) C[i] acc'[i]

def attention_like [q][m][n][k] (A: [m][k]f16) (B: [q][k][n]f16) : [m][n]f32 =
  -- Copy to shared
  let A' = if q > 1
           then copy A
           else replicate (m * k) 0.0f16 |> unflatten

  let acc_init : *[m][n]f32 = replicate (m * n) 0.0f32 |> unflatten in
  --let acc_init : *[m][k]f32 = replicate (m * k) 0.0f32 |> unflatten in
  loop _ = (acc_init: *[m][n]f32) for i < q do
    let B' = B[i]
    let C : *[m][n]f32 = matmulf32 A' B'
    in copy C

entry mk_input (p: i64) (q: i64) (m: i64) (n: i64) (k: i64) =
  let A = replicate (p * m * k) 1.0f16 |> unflatten_3d
  let B = replicate (p * q * k * n) 1.0f16 |> unflatten_4d
  in (A, B)

entry run_no_intra [q][p][d] (A: [p][d][d]f16) (B: [p][q][d][d]f16) =
  map2 attention_like A B

entry run16 [q][p] (A: [p][16][16]f16) (B: [p][q][16][16]f16) =
  #[incremental_flattening(only_intra)]map2 attention_like A B

entry run32 [q][p] (A: [p][32][32]f16) (B: [p][q][32][32]f16) =
  #[incremental_flattening(only_intra)]map2 attention_like A B
   
entry run64 [q][p] (A: [p][64][64]f16) (B: [p][q][64][64]f16) =
  #[incremental_flattening(only_intra)]map2 attention_like A B

entry run128 [q][p] (A: [p][128][128]f16) (B: [p][q][128][128]f16) =
  #[incremental_flattening(only_intra)]map2 attention_like A B    
