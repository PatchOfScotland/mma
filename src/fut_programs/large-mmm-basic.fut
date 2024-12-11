-- ==
-- compiled random input {[1024][1024]f16 [1024][1024]f16}
-- compiled random input {[2048][2048]f16 [2048][2048]f16}
-- compiled random input {[4096][4096]f16 [4096][4096]f16}
-- compiled random input {[2048][1024]f16 [1024][2048]f16}
-- compiled random input {[4096][2048]f16 [2048][4096]f16}
-- compiled random input {[8192][2048]f16 [2048][8192]f16}

import "mmm-helpers"

entry main [m][n][k] (A: [m][k]f16) (B: [k][n]f16) : [m][n]f32 =
  matmulf32 A B

-- large-mmm-basic.fut (using large-mmm-basic.fut.tuning):
-- [2048][1024]f16 [1024][2048]f16:       1468μs (95% CI: [    1467.5,     1469.5])
-- [4096][2048]f16 [2048][4096]f16:      11398μs (95% CI: [   11396.5,    11398.5])
-- [4096][4096]f16 [4096][4096]f16:      23093μs (95% CI: [   22786.9,    23204.3])
                                                          
