-- ==
-- compiled random input {[4096][2048]f16 [2048][4096]f16}

import "mmm-helpers"

entry main [m][n][k] (A: [m][k]f16) (B: [k][n]f16) : [m][n]f16 =
  matmulf16 A B
