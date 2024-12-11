-- ==
-- entry: lud64
-- compiled random input {[256][64][64]f16 [256][64][64]f16 [256][256][64][64]f32}

import "lud-mmm"
entry lud64 [m] (top_per: [m][64][64]f16)
         (lft_per: [m][64][64]f16)
         (mat_slice: [m][m][64][64]f32 )
          =
  ludMult 64 (top_per, lft_per, mat_slice)
