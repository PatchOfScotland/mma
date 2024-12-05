import "mmm-helpers"

def mmm_intra [q] [d] (A: [q][d][d]f16) (B: [q][d][d]f16) : [q][d][d]f32 =
  #[incremental_flattening(only_intra)]map2 matmulf32 A B

entry mmm_intra16 [q] (A: [q][16][16]f16) (B: [q][16][16]f16) =
  mmm_intra A B

entry mmm_intra32 [q] (A: [q][32][32]f16) (B: [q][32][32]f16) =
  mmm_intra A B

entry mmm_intra64 [q] (A: [q][64][64]f16) (B: [q][64][64]f16) =
  mmm_intra A B

