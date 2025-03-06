-- ==
-- entry: lud16
-- compiled random input {[256][16][16]f16 [256][16][16]f16 [256][256][16][16]f32}

-- ==
-- entry: lud32
-- compiled random input {[256][32][32]f16 [256][32][32]f16 [256][256][32][32]f32}

-- ==
-- entry: lud64
-- compiled random input {[256][64][64]f16 [256][64][64]f16 [256][256][64][64]f32}

-- ==
-- entry: lud128
-- compiled random input {[256][128][128]f16 [256][128][128]f16 [256][256][128][128]f32}


import "mmm-helpers"
import "seq_acc"

#[inline]
def ludMult [m][b] (r: i64) (top_per: [m][b][b]f16, lft_per: [m][b][b]f16, mat_slice: [m][m][b][b]f32) =
  #[incremental_flattening(only_inner)]
  map (\(mat_arr: [m][b][b]f32, lft: [b][b]f16)  ->
         #[incremental_flattening(only_intra)]
         map (\(mat_blk: [b][b]f32, top: [b][b]f16)  ->
                let mm = matmulf32 lft top
                -- in mm
                in seq_acc r (+) (copy mat_blk) (copy mm)
             ) (zip mat_arr top_per)
      ) (zip mat_slice lft_per)

entry lud16 [m] (top_per: [m][16][16]f16)
         (lft_per: [m][16][16]f16)
         (mat_slice: [m][m][16][16]f32 )
          =
  ludMult 16 (top_per, lft_per, mat_slice)
                              
entry lud32 [m] (top_per: [m][32][32]f16)
         (lft_per: [m][32][32]f16)
         (mat_slice: [m][m][32][32]f32 )
          =
  ludMult 32 (top_per, lft_per, mat_slice)
                              
entry lud64 [m] (top_per: [m][64][64]f16)
         (lft_per: [m][64][64]f16)
         (mat_slice: [m][m][64][64]f32 )
          =
  ludMult 64 (top_per, lft_per, mat_slice)
                              
entry lud128 [m] (top_per: [m][128][128]f16)
         (lft_per: [m][128][128]f16)
         (mat_slice: [m][m][128][128]f32 )
          =
  ludMult 128 (top_per, lft_per, mat_slice)          

