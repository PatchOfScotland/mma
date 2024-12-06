-- ==
-- entry: run_no_intra
-- compiled random input {[1024][16][16]f16 [1024][256][16][16]f16}
-- compiled random input {[1024][32][32]f16 [1024][256][32][32]f16}
-- compiled random input {[1024][64][64]f16 [1024][256][64][64]f16}
-- compiled random input {[1024][128][128]f16 [1024][256][128][128]f16}

import "attention_like"

entry run_no_intra [q][p][d] (A: [p][d][d]f16) (B: [p][q][d][d]f16) =
  map2 attention_like A B
