type real = f32
def real_exp = f16.exp
def real_i64 = f16.i64
def real_max = f16.max
def real_lowest = f16.lowest
def real_sqrt = f16.sqrt

def imap xs f = map f xs
def imap3 xs ys zs f = map3 f xs ys zs

def chunk : i64 = 1024
def fseq  : i64 = 32

def copy2shr [n] (xs: [n]f16) : *[n]f16 = #[unsafe]
  let xs' = copy xs
  in  if opaque(true) then xs'
      else xs' with [0] = 0f16

def reduceEffSeq [g] (f: f16 -> f16) (bop: f16->f16->f16) (ne: f16) (xs: [g*fseq]f16) : f16 = #[unsafe]
  let redPerThd (tid: i64) =
    loop r = ne for i < fseq do
        bop r (f xs[i*g + tid])
  -- per-thread reduce
  let rs = map redPerThd (iota g)
  in  reduce_comm bop ne rs

def softmaxChunkML (q: i64) (xs_glb: [q*chunk]f16) : (f16,f16) = #[unsafe]
  let g = chunk/fseq in
  loop (mi_old : f16, li_old : f16) = (real_lowest, 0.0)
  for i < q do
    let xs = copy2shr ( xs_glb[i*chunk: i*chunk + chunk] :> [g*fseq]f16 )
    let xs = xs
    --
    let maxi = reduceEffSeq id real_max real_lowest xs
    let sumi = reduceEffSeq (\x -> real_exp (x - maxi)) (+) 0.0 xs
    --
    let mi_new = real_max mi_old maxi
    let eij = real_exp (maxi - mi_new)
    let eli = li_old * (f16.exp (mi_old - mi_new))
    let li_new = eli + sumi * eij
    in  (mi_new, li_new)
    -- this saves one f16.exp operation:
--    let exp_term = real_exp (mi_old - maxi)
--    in  if mi_old < maxi
--        then ( maxi,   li_old * exp_term + sumi )
--        else ( mi_old, li_old + sumi / exp_term )  

def softmaxOnline [m][n] (xss: [m][n]f16) : [m][n]f16 = #[unsafe]
  let q = assert (n % chunk == 0) (n / chunk)
  let (mis, lis) = 
          #[incremental_flattening(only_intra)]
          map (softmaxChunkML q) (xss :> [m][q*chunk]f16)
      |> unzip |> opaque
  in  imap3 xss mis lis
        (\ xs mi li ->
          map (\ x -> real_exp (x - mi) / li ) xs
        ) |> opaque

def softmax [m][n] (xss: [m][n]f16) : [m][n]f16 =
  let f xs =
    let max = reduce real_max real_lowest xs
    let xs' = map (\x -> real_exp (x - max)) xs
    let weight = 1.0 / reduce (+) 0.0 xs'
    in  map (* weight) xs'
  in map f xss

def matmulT [m][n][k] (a: [m][k]f16) (b: [n][k]f16) : [m][n]f16 =
  imap a (\a_row -> imap b (\b_row -> map2 (*) a_row b_row |> reduce (+) 0.0) )

def matmul [m][n][k] (a: [m][k]f16) (b: [k][n]f16) : [m][n]f16 =
  matmulT a (transpose b)

def matmul_batched [n] (A: [n][n]f16) (B: [n][n]f16) : [n][n]f16 =
    let c = map (\Arow ->
        map (\Bcol ->
            map2 (*) Arow Bcol
                |> map f32.f16
                |> reduce (+) 0f32
                |> f16.f32 
            ) (B)
        ) (A)
    in (transpose c)

def make_batches [md][d] (k:[md][d]f16) (n:i64): [n][d][d]f16 =
  let _ = assert (md % d == 0) (1) --Check that md divides neatly by d 
  let splits: *[n][d][d]f16 =
    map (\h: [d][d]f16  ->
        map (\i: [d]f16  ->
          map (\j: f16  ->
            #[unsafe] k[h*d+i, j]
          ) (iota d)
        ) (iota d)
      ) (iota n)
  in (splits)

def combine [j] (i:i64) (n:i64) (m:[n][j][j]f16): [j][i]f16 =
    let combined: [i][j]f16 = 
        map (\di -> 
            map (\dj -> 
                #[unsafe] m[di/j,di%j,dj]
            ) (iota j)
        ) (iota (i))
    in (transpose combined)

def matmul_split [d] (n: i64) (Qi:[d][d]f16) (K:[n*d][d]f16): [d][n*d]f16 =
  let splits = make_batches K n  --[n][d][d]f16
  let mmm = #[incremental_flattening(only_intra)]map (matmul_batched (transpose Qi)) (splits) --[n][j][j]f16
  let i = n * d
  let combined = combine i n mmm --[j][i]f16
  in (combined)

def oneIter [d] (m: i64) (K: [m*d][d]f16) (V: [m*d][d]f16) (Qi: [d][d]f16) : [d][d]f16 =
  let P_block = matmul_split m Qi K |> opaque -- : [d][m*d]f16
  -- let P_block = softmax P_block
  let P_block = softmaxOnline P_block  -- : [d][m*d]f16
  in  matmul P_block V      -- : [d][d]f16

def FlashAttention [d][m] 
        (Q: [m][d][d]f16) 
        (K: [m*d][d]f16) 
        (V: [m*d][d]f16) 
      : [m][d][d]f16 =
  map (oneIter m K V) Q

def L2 [n] (xs: [n]f16) : f16 =
    map (\x -> x*x) xs
    |> reduce (+) 0.0
    |> real_sqrt  
  
entry mk_input (m:i64) (d:i64) : ([m][d][d]f16, [m*d][d]f16, [m*d][d]f16) =
  let Q = replicate d 1.0 |> replicate d |> replicate m
  let K = replicate d 1.0 |> replicate (m*d)
  let V = replicate d 1.0 |> replicate (m*d)
  in  (Q, K, V)

--
-- ==
-- entry: main64
-- "Class 8192-64 " script input { (mk_input 128i64 64i64) }
-- "Class 16384-64 " script input { (mk_input 256i64 64i64) }
-- "Class 32768-64 " script input { (mk_input 512i64 64i64) }

-- "Class 65536-64 " script input { (mk_input 1024i64 64i64) }

entry main64 [m] (Q: [m][64][64]f16) (K: [m*64][64]f16) (V: [m*64][64]f16) =
  FlashAttention Q K V

--
-- ==
-- entry: main128
-- "Class 64-128 " script input { (mk_input 64i64 128i64) }
-- "Class 128-128" script input { (mk_input 128i64 128i64) }
-- "Class 256-128" script input { (mk_input 256i64 128i64) }

-- "Class 512-128" script input { (mk_input 512i64 128i64) }

entry main128 [m] (Q: [m][128][128]f16) (K: [m*128][128]f16) (V: [m*128][128]f16) =
  FlashAttention Q K V


--
-- ==
-- entry: thesislike
-- "Class 8192-64 " script input { (mk_input 128i64 64i64) }
-- "Class 8192-128" script input { (mk_input 64i64 128i64) }
entry thesislike [m][d] (Q: [m][d][d]f16) (K: [m*d][d]f16) (V: [m*d][d]f16) =
  FlashAttention Q K V

--
-- ==
-- entry: validate
-- "Class 16384-64 " nobench script input { (mk_input 256i64 64i64) }
-- output { 0.0f32 }
-- "Class 32768-64 " nobench script input { (mk_input 512i64 64i64) }
-- output { 0.0f32 }
-- "Class 8192-128 " nobench script input { (mk_input 64i64 128i64) }
-- output { 0.0f32 }
-- "Class 16384-128" nobench script input { (mk_input 128i64 128i64) }
-- output { 0.0f32 }

entry validate [m][d] (Q: [m][d][d]f16) (K: [m*d][d]f16) (V: [m*d][d]f16): f16 =
  let O = FlashAttention Q K V
  let O_flat = flatten (flatten O)
  in ( L2 O_flat ) - (real_sqrt (real_i64 (m*d*d)))
  -- Denoting with N = m*d, 
  -- THE NUMBER OF FLOPS IS: 4 * d * N * N
  -- ALSO, Datasets are named "Class N-d"