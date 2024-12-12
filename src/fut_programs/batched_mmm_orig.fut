-- ==
-- entry: mmmf16
-- compiled random input { [8192][16][16]f16 [8192][16][16]f16 }
-- compiled random input { [8192][32][32]f16 [8192][32][32]f16 }
-- compiled random input { [8192][64][64]f16 [8192][64][64]f16 }
-- compiled random input { [8192][128][128]f16 [8192][128][128]f16 }

-- ==
-- entry: mmmf32
-- compiled random input { [8192][16][16]f32 [8192][16][16]f32 }
-- compiled random input { [8192][32][32]f32 [8192][32][32]f32 }
-- compiled random input { [8192][64][64]f32 [8192][64][64]f32 }
-- compiled random input { [8192][128][128]f32 [8192][128][128]f32 }

import "mmm-helpers"                

entry mmmf16 [q] [d] (A: [q][d][d]f16) (B: [q][d][d]f16) : [q][d][d]f32 =
  map2 matmulf16 A B

entry mmmf32 [q] [d] (A: [q][d][d]f16) (B: [q][d][d]f16) : [q][d][d]f32 =
  map2 mmm_no_intra_f32 A B
