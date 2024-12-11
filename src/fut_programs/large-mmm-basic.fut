-- ==
-- entry: mmm_f16
-- compiled random input {[1024][1024]f16 [1024][1024]f16}
-- compiled random input {[2048][2048]f16 [2048][2048]f16}
-- compiled random input {[4096][4096]f16 [4096][4096]f16}
-- compiled random input {[2048][1024]f16 [1024][2048]f16}
-- compiled random input {[4096][2048]f16 [2048][4096]f16}
-- compiled random input {[8192][2048]f16 [2048][8192]f16}

-- ==
-- entry: mmm_f32
-- compiled random input {[1024][1024]f32 [1024][1024]f32}
-- compiled random input {[2048][2048]f32 [2048][2048]f32}
-- compiled random input {[4096][4096]f32 [4096][4096]f32}
-- compiled random input {[2048][1024]f32 [1024][2048]f32}
-- compiled random input {[4096][2048]f32 [2048][4096]f32}
-- compiled random input {[8192][2048]f32 [2048][8192]f32}

import "mmm-helpers"

entry mmm_f16 [m][n][k] (A: [m][k]f16) (B: [k][n]f16) : [m][n]f16 =
  matmulf16 A B

entry mmm_f32 [m][n][k] (A: [m][k]f32) (B: [k][n]f32) : [m][n]f32 =
  mmm_no_intra_f32 A B                                                             

                                                          

-- [1024][1024]f16 [1024][1024]f16:        439μs (95% CI: [     438.9,      439.2])
-- [2048][2048]f16 [2048][2048]f16:       3068μs (95% CI: [    3062.2,     3072.8])
-- [4096][4096]f16 [4096][4096]f16:      23967μs (95% CI: [   23946.3,    23983.4])
-- [2048][1024]f16 [1024][2048]f16:       1557μs (95% CI: [    1555.4,     1558.3])
-- [4096][2048]f16 [2048][4096]f16:      12614μs (95% CI: [   12601.2,    12619.8])
-- [8192][2048]f16 [2048][8192]f16:      50299μs (95% CI: [   50113.2,    50459.6])
