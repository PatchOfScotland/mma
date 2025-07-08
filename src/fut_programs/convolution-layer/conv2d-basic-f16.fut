-- Taken from https://github.com/PeterLarsen404/Diffusion/blob/master/layers/conv2d.fut

def dotprod [n] (xs: [n]f16) (ys: [n]f16): f16 =
    (reduce (+) (0f16) (map2 (*) xs ys))

def matmul [n][p][m] (xss: [n][p]f16) (yss: [p][m]f16): *[n][m]f16 =
    map (\xs -> map (dotprod xs) (transpose yss)) xss

def add_padding [l][n][m] (imgs : [l][n][m]f16) (padding : i64) =
  let n_pad = (n+(padding*2))
  let m_pad = (m+(padding*2))
  in map (\ img_i -> tabulate_2d n_pad m_pad (\ i j ->
      if (i < padding || i >= (n+padding) || j < padding || j >= (m+padding)) then 0 else img_i[i-padding,j-padding])) imgs

def im2col [l][n][m] (img : [l][n][m]f16) (total : i64) (kernel_size : i64) (new_n : i64) (new_m : i64) =
  let k_total = kernel_size*kernel_size
  in transpose (flatten (tabulate_2d new_n new_m (\ y x -> flatten (map (\ i -> flatten (i[y:y+kernel_size, x:x+kernel_size]) :> [k_total]f16) img) :> [total]f16)))


def convolve2D [n][m][p][k][l][o] (imgs : [l][n][m]f16) (kernels : [o][l][p][k]f16) (biases : [o]f16) (padding : i64) =
  let new_n = (((n+(padding*2))-p)+1)
  let new_m = (((m+(padding*2))-p)+1)
  let total = l*p*k

  let imgs_padded =
    if (padding != 0) then
      add_padding imgs padding
    else
      imgs

  let img_col = im2col imgs_padded total p new_n new_m --[total][new_n*new_m]f16
  let kernel_col = map (\ x -> flatten_3d x :> [total]f16) kernels --[o][total]f16
  let res = matmul kernel_col img_col --[o][new_n*new_m]f16
  let res_bias = map2 (\ r b -> map (+b) r) res biases --[o][new_n*new_m]f16
  in map (unflatten) res_bias

-- Bench conv2d
-- ==
-- entry: convolve2d_test_img_size
-- nobench input @data/c_1024-128-128f16_256-1024-3-3f16_256f16_1i64.in output @data/c_1024-128-128f16_256-1024-3-3f16_256f16_1i64_basic.out
-- "m32_n32_p3_k3_l64_o16_pad1" compiled random input {[64][32][32]f16 [16][64][3][3]f16 [16]f16 1i64}
-- "m128_n128_p3_k3_l1024_o256_pad1" compiled random input {[1024][128][128]f16 [256][1024][3][3]f16 [256]f16 1i64}
-- "m256_n256_p3_k3_l1024_o256_pad1" compiled random input {[1024][256][256]f16 [256][1024][3][3]f16 [256]f16 1i64}
-- "m512_n512_p3_k3_l1024_o256_pad1" compiled random input {[1024][512][512]f16 [256][1024][3][3]f16 [256]f16 1i64}

-- Bench conv2d
-- ==
-- entry: convolve2d_test_input


-- "m256_n256_p3_k3_l1024_o256_pad1" compiled random input {[1024][256][256]f16 [256][1024][3][3]f16 [256]f16 1i64}
-- "m256_n256_p3_k3_l2048_o256_pad1" compiled random input {[2048][256][256]f16 [256][2048][3][3]f16 [256]f16 1i64}
-- "m256_n256_p3_k3_l4096_o256_pad1" compiled random input {[4096][256][256]f16 [256][4096][3][3]f16 [256]f16 1i64}

-- Bench conv2d
-- ==
-- entry: convolve2d_test_output


-- "m256_n256_p3_k3_l1024_o256_pad1" compiled random input {[1024][256][256]f16 [256][1024][3][3]f16 [256]f16 1i64}
-- "m256_n256_p3_k3_l1024_o512_pad1" compiled random input {[1024][256][256]f16 [512][1024][3][3]f16 [512]f16 1i64}
-- "m256_n256_p3_k3_l1024_o1024_pad1" compiled random input {[1024][256][256]f16 [1024][1024][3][3]f16 [1024]f16 1i64}
-- "m256_n256_p3_k3_l1024_o2048_pad1" compiled random input {[1024][256][256]f16 [2048][1024][3][3]f16 [2048]f16 1i64}
-- "m256_n256_p3_k3_l1024_o4096_pad1" compiled random input {[1024][256][256]f16 [4096][1024][3][3]f16 [4096]f16 1i64}
-- "m256_n256_p3_k3_l1024_o8192_pad1" compiled random input {[1024][256][256]f16 [8192][1024][3][3]f16 [8192]f16 1i64}

-- compiled random input {[1024][1024][1024]f16 [256][1024][3][3]f16 [256]f16 1i64}

-- compiled random input {[1024][16][16]f16 [256][1024][3][3]f16 [256]f16 1i64}
-- compiled random input {[1024][32][32]f16 [256][1024][3][3]f16 [256]f16 1i64}
-- compiled random input {[1024][64][64]f16 [256][1024][3][3]f16 [256]f16 1i64}

-- compiled random input {[1][28][28]f16 [64][1][3][3]f16 [64]f16 1i64}
-- compiled random input {[512][28][28]f16 [128][512][3][3]f16 [128]f16 1i64}
-- compiled random input {[1024][28][28]f16 [256][1024][3][3]f16 [256]f16 1i64}
-- compiled random input {[1][64][64]f16 [1][1][3][3]f16 [1]f16 1i64}
-- compiled random input {[1][64][64]f16 [1][1][7][7]f16 [1]f16 1i64}
-- compiled random input {[1][512][512]f16 [1][1][3][3]f16 [1]f16 1i64}
-- compiled random input {[1][1024][1024]f16 [1][1][3][3]f16 [1]f16 1i64}
-- compiled random input {[1][1][1]f16 [1024][1][3][3]f16 [1024]f16 1i64}
-- compiled random input {[1024][1][1]f16 [1][1024][3][3]f16 [1]f16 1i64}
-- compiled random input {[50][50][50]f16 [100][50][3][3]f16 [100]f16 1i64}
-- compiled random input {[200][50][50]f16 [400][200][3][3]f16 [400]f16 1i64}
-- compiled random input {[500][100][100]f16 [1000][500][3][3]f16 [1000]f16 1i64}


entry convolve2d_test_img_size [n][m][p][k][l][o] (imgs : [l][n][m]f16) (kernels : [o][l][p][k]f16) (biases : [o]f16) (padding : i64) =
  convolve2D imgs kernels biases padding

entry convolve2d_test_input [n][m][p][k][l][o] (imgs : [l][n][m]f16) (kernels : [o][l][p][k]f16) (biases : [o]f16) (padding : i64) =
  convolve2D imgs kernels biases padding

entry convolve2d_test_output [n][m][p][k][l][o] (imgs : [l][n][m]f16) (kernels : [o][l][p][k]f16) (biases : [o]f16) (padding : i64) =
  convolve2D imgs kernels biases padding