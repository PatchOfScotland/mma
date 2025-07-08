-- Taken from https://github.com/PeterLarsen404/Diffusion/blob/master/layers/conv2d.fut

def dotprod [n] (xs: [n]f32) (ys: [n]f32): f32 =
    (reduce (+) (0f32) (map2 (*) xs ys))

def matmul [n][p][m] (xss: [n][p]f32) (yss: [p][m]f32): *[n][m]f32 =
    map (\xs -> map (dotprod xs) (transpose yss)) xss

def add_padding [l][n][m] (imgs : [l][n][m]f32) (padding : i64) =
  let n_pad = (n+(padding*2))
  let m_pad = (m+(padding*2))
  in map (\ img_i -> tabulate_2d n_pad m_pad (\ i j ->
      if (i < padding || i >= (n+padding) || j < padding || j >= (m+padding)) then 0 else img_i[i-padding,j-padding])) imgs

def im2col [l][n][m] (img : [l][n][m]f32) (total : i64) (kernel_size : i64) (new_n : i64) (new_m : i64) =
  let k_total = kernel_size*kernel_size
  in transpose (flatten (tabulate_2d new_n new_m (\ y x -> flatten (map (\ i -> flatten (i[y:y+kernel_size, x:x+kernel_size]) :> [k_total]f32) img) :> [total]f32)))

def convolve2D [n][m][p][k][l][o] (imgs : [l][n][m]f32) (kernels : [o][l][p][k]f32) (biases : [o]f32) (padding : i64) =
  let new_n = (((n+(padding*2))-p)+1)
  let new_m = (((m+(padding*2))-p)+1)
  let total = l*p*k

  let imgs_padded =
    if (padding != 0) then
      add_padding imgs padding
    else
      imgs

  let img_col = im2col imgs_padded total p new_n new_m --[total][new_n*new_m]f32
  let kernel_col = map (\ x -> flatten_3d x :> [total]f32) kernels --[o][total]f32
  let res = matmul kernel_col img_col --[o][new_n*new_m]f32
  let res_bias = map2 (\ r b -> map (+b) r) res biases --[o][new_n*new_m]f32
  in map (unflatten) res_bias

-- Bench conv2d
-- ==
-- entry: convolve2d_test_img_size
-- "m128_n128_p3_k3_l1024_o256_pad1" compiled random input {[1024][128][128]f32 [256][1024][3][3]f32 [256]f32 1i64}
-- "m256_n256_p3_k3_l1024_o256_pad1" compiled random input {[1024][256][256]f32 [256][1024][3][3]f32 [256]f32 1i64}
-- "m512_n512_p3_k3_l1024_o256_pad1" compiled random input {[1024][512][512]f32 [256][1024][3][3]f32 [256]f32 1i64}

-- Bench conv2d
-- ==
-- entry: convolve2d_test_input
-- "m256_n256_p3_k3_l1024_o256_pad1" compiled random input {[1024][256][256]f32 [256][1024][3][3]f32 [256]f32 1i64}
-- "m256_n256_p3_k3_l2048_o256_pad1" compiled random input {[2048][256][256]f32 [256][2048][3][3]f32 [256]f32 1i64}
-- "m256_n256_p3_k3_l4096_o256_pad1" compiled random input {[4096][256][256]f32 [256][4096][3][3]f32 [256]f32 1i64}

-- Bench conv2d
-- ==
-- entry: convolve2d_test_output
-- "m256_n256_p3_k3_l1024_o256_pad1" compiled random input {[1024][256][256]f32 [256][1024][3][3]f32 [256]f32 1i64}
-- "m256_n256_p3_k3_l1024_o512_pad1" compiled random input {[1024][256][256]f32 [512][1024][3][3]f32 [512]f32 1i64}
-- "m256_n256_p3_k3_l1024_o1024_pad1" compiled random input {[1024][256][256]f32 [1024][1024][3][3]f32 [1024]f32 1i64}
-- "m256_n256_p3_k3_l1024_o2048_pad1" compiled random input {[1024][256][256]f32 [2048][1024][3][3]f32 [2048]f32 1i64}
-- "m256_n256_p3_k3_l1024_o4096_pad1" compiled random input {[1024][256][256]f32 [4096][1024][3][3]f32 [4096]f32 1i64}
-- "m256_n256_p3_k3_l1024_o8192_pad1" compiled random input {[1024][256][256]f32 [8192][1024][3][3]f32 [8192]f32 1i64}

-- compiled random input {[1024][1024][1024]f32 [256][1024][3][3]f32 [256]f32 1i64}

-- compiled random input {[1024][16][16]f32 [256][1024][3][3]f32 [256]f32 1i64}
-- compiled random input {[1024][32][32]f32 [256][1024][3][3]f32 [256]f32 1i64}
-- compiled random input {[1024][64][64]f32 [256][1024][3][3]f32 [256]f32 1i64}

-- compiled random input {[1][28][28]f32 [64][1][3][3]f32 [64]f32 1i64}
-- compiled random input {[512][28][28]f32 [128][512][3][3]f32 [128]f32 1i64}
-- compiled random input {[1024][28][28]f32 [256][1024][3][3]f32 [256]f32 1i64}
-- compiled random input {[1][64][64]f32 [1][1][3][3]f32 [1]f32 1i64}
-- compiled random input {[1][64][64]f32 [1][1][7][7]f32 [1]f32 1i64}
-- compiled random input {[1][512][512]f32 [1][1][3][3]f32 [1]f32 1i64}
-- compiled random input {[1][1024][1024]f32 [1][1][3][3]f32 [1]f32 1i64}
-- compiled random input {[1][1][1]f32 [1024][1][3][3]f32 [1024]f32 1i64}
-- compiled random input {[1024][1][1]f32 [1][1024][3][3]f32 [1]f32 1i64}
-- compiled random input {[50][50][50]f32 [100][50][3][3]f32 [100]f32 1i64}
-- compiled random input {[200][50][50]f32 [400][200][3][3]f32 [400]f32 1i64}
-- compiled random input {[500][100][100]f32 [1000][500][3][3]f32 [1000]f32 1i64}

entry convolve2d_test_img_size [n][m][p][k][l][o] (imgs : [l][n][m]f32) (kernels : [o][l][p][k]f32) (biases : [o]f32) (padding : i64) =
  convolve2D imgs kernels biases padding

entry convolve2d_test_input [n][m][p][k][l][o] (imgs : [l][n][m]f32) (kernels : [o][l][p][k]f32) (biases : [o]f32) (padding : i64) =
  convolve2D imgs kernels biases padding

entry convolve2d_test_output [n][m][p][k][l][o] (imgs : [l][n][m]f32) (kernels : [o][l][p][k]f32) (biases : [o]f32) (padding : i64) =
  convolve2D imgs kernels biases padding
