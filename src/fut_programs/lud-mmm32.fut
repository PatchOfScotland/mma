-- ==
-- entry: lud32
-- compiled random input {[128][32][32]f16 [128][32][32]f16 [128][128][32][32]f32}

import "lud-mmm"
entry lud32 [m] (top_per: [m][32][32]f16)
         (lft_per: [m][32][32]f16)
         (mat_slice: [m][m][32][32]f32 )
          =
  ludMult 32 (top_per, lft_per, mat_slice)
