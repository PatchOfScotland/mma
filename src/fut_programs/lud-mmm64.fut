-- ==
-- entry: lud64
-- compiled random input {[128][64][64]f16 [128][64][64]f16 [128][128][64][64]f32}

import "lud-mmm"
entry lud64 [m] (top_per: [m][64][64]f16)
         (lft_per: [m][64][64]f16)
         (mat_slice: [m][m][64][64]f32 )
          =
  ludMult 32 (top_per, lft_per, mat_slice)
