-- ==
-- entry: mmm_intra16
-- compiled random input { [32768][16][16]f16 [32768][16][16]f16 }

-- ==
-- entry: mmm_intra32
-- compiled random input { [32768][32][32]f16 [32768][32][32]f16 }

-- ==
-- entry: mmm_intra64
-- compiled random input { [32768][64][64]f16 [32768][64][64]f16 }

-- ==
-- entry: mmm_intra128
-- compiled random input { [32768][128][128]f16 [32768][128][128]f16 }

import "mmm-helpers"

entry mk_input (q: i64) (m: i64) (n: i64) (k: i64) =
  let A = replicate (q * m * k) 1.0f16 |> unflatten_3d
  let B = replicate (q * q * k * n) 1.0f16 |> unflatten_3d
  in (A, B)                                      

def mmm_intra [q] [d] (A: [q][d][d]f16) (B: [q][d][d]f16) : [q][d][d]f32 =
  #[incremental_flattening(only_intra)]map2 matmulf32 A B

entry mmm_intra16 [q] (A: [q][16][16]f16) (B: [q][16][16]f16) =
  mmm_intra A B

entry mmm_intra32 [q] (A: [q][32][32]f16) (B: [q][32][32]f16) =
  mmm_intra A B

entry mmm_intra64 [q] (A: [q][64][64]f16) (B: [q][64][64]f16) =
  mmm_intra A B

entry mmm_intra128 [q] (A: [q][128][128]f16) (B: [q][128][128]f16) =
  mmm_intra A B                                                            
