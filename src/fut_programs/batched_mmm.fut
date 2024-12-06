-- ==
-- entry: mmm_intra16
-- only_intra compiled random input {[10000][16][16]f16 [10000][16][16]f16} auto output

-- ==
-- entry: mmm_intra32
-- only_intra compiled random input {[10000][32][32]f16 [10000][32][32]f16} auto output

-- ==
-- entry: mmm_intra64
-- only_intra compiled random input {[10000][64][64]f16 [10000][64][64]f16} auto output

-- ==
-- entry: mmm_intra128
-- only_intra compiled random input {[10000][128][128]f16 [10000][128][128]f16} auto output

import "mmm-helpers"

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
