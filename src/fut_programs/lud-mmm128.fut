-- ==
-- entry: lud128
-- compiled random input {[128][128][128]f16 [128][128][128]f16 [128][128][128][128]f32}

import "lud-mmm"

entry lud128 [m] (top_per: [m][128][128]f16)
         (lft_per: [m][128][128]f16)
         (mat_slice: [m][m][128][128]f32 )
          =
  ludMult 128 (top_per, lft_per, mat_slice)          


