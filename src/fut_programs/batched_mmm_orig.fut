-- ==
-- entry: mmm
-- compiled random input {[100000][16][16]f16 [100000][16][16]f16} auto output
-- compiled random input {[100000][32][32]f16 [100000][32][32]f16} auto output
-- compiled random input {[100000][64][64]f16 [100000][64][64]f16} auto output
-- compiled random input {[100000][128][128]f16 [100000][128][128]f16} auto output

import "batched_mmm"
import "mmm-helpers"       

entry mmm [q] [d] (A: [q][d][d]f16) (B: [q][d][d]f16) : [q][d][d]f32 =
  map2 matmulf32 A B
                   
