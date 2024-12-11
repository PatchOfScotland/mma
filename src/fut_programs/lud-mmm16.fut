-- ==
-- entry: lud16
-- compiled random input {[256][16][16]f16 [256][16][16]f16 [256][256][16][16]f32}


import "lud-mmm"

entry lud16 [m] (top_per: [m][16][16]f16)
         (lft_per: [m][16][16]f16)
         (mat_slice: [m][m][16][16]f32 )
          =
  ludMult 16 (top_per, lft_per, mat_slice)
