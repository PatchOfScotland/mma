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

def unflatten_to 't [k] (n: i64) (m: i64) (A: [k]t) : [n][m]t =
  unflatten (A :> [n * m]t)

def combine [d] (m:i64) (A:[m][d][d]f16): [d][m*d]f16 =
    transpose (flatten A)

def matmul_split_batched [d] (m: i64) (Qi:[d][d]f16) (K:[m*d][d]f16): [d][m*d]f16 =
  let splits = unflatten_to m d K  --[n][d][d]f16
  let s=64i64
  let batches = unflatten_to (m/s) s splits
  let mmm_nf = map (#[incremental_flattening(only_intra)]map (matmulT Qi)) batches --[m/s][s][d][d]f16
  let mmm = (#[incremental_flattening(only_intra)]map (\i -> mmm_nf[i/s][i%s]) (iota (m))) --[m][d][d]f16
      --map (matmul_batched (Qi)) (splits)
  let combined = combine m mmm --[j][i]f16
  in (combined)

def matmul_split [d] (m: i64) (Qi:[d][d]f16) (K:[m*d][d]f16): [d][m*d]f16 =
  combine m (map (matmul_batched (Qi)) (unflatten_to m d K)) --[j][i]f16

def oneIterBatched [d] (m: i64) (K: [m*d][d]f16) (V: [m*d][d]f16) (Qi: [d][d]f16) : [d][m*d]f16 =
  let P_block = matmul_split_batched m Qi K |> opaque -- : [d][m*d]f16
  let P_block = softmaxOnline P_block  -- : [d][m*d]f16
  in P_block
  -- Too large for current implementation
  --in matmul P_block V      -- : [d][d]f16

def oneIter [d] (m: i64) (K: [m*d][d]f16) (V: [m*d][d]f16) (Qi: [d][d]f16) : [d][d]f16 =
  let P_block = #[incremental_flattening(only_intra)]matmul_split m Qi K |> opaque -- : [d][m*d]f16
  let P_block = softmaxOnline P_block  -- : [d][m*d]f16
  in #[incremental_flattening(only_intra)]matmul P_block V      -- : [d][d]f16
  
def FlashAttentionBatched [d][m] 
        (Q: [m][d][d]f16) 
        (K: [m*d][d]f16) 
        (V: [m*d][d]f16) 
      : [m][d][m*d]f16 =
  map (oneIterBatched m K V) Q

def FlashAttention [d][m] 
        (Q: [m][d][d]f16) 
        (K: [m*d][d]f16) 
        (V: [m*d][d]f16) 
      : [m][d][d]f16 =
  map (oneIter m K V) Q
  
entry mk_input (m:i64) (d:i64): ([m][d][d]f16, [m*d][d]f16, [m*d][d]f16) =
  let Q = replicate d 1.0 |> replicate d |> replicate m
  let K = replicate d 1.0 |> replicate (m*d)
  let V = replicate d 1.0 |> replicate (m*d)
  in  (Q, K, V)

--
-- ==
-- entry: thesislike16
-- "Class 8192-16 " script input { (mk_input 512i64 16i64) }

entry thesislike16 [m][d] (Q: [m][d][d]f16) (K: [m*d][d]f16) (V: [m*d][d]f16) =
  FlashAttentionBatched Q K V

--
-- ==
-- entry: thesislike32
-- "Class 8192-32 " script input { (mk_input 256i64 32i64) }

entry thesislike32 [m][d] (Q: [m][d][d]f16) (K: [m*d][d]f16) (V: [m*d][d]f16) =
  FlashAttentionBatched Q K V

--
-- ==
-- entry: thesislike64
-- "Class 8192-64 " script input { (mk_input 128i64 64i64) }

entry thesislike64 [m][d] (Q: [m][d][d]f16) (K: [m*d][d]f16) (V: [m*d][d]f16) =
  FlashAttention Q K V

--
-- ==
-- entry: thesislike128
-- "Class 8192-128" script input { (mk_input 64i64 128i64) }

entry thesislike128 [m][d] (Q: [m][d][d]f16) (K: [m*d][d]f16) (V: [m*d][d]f16) =
  FlashAttention Q K V

--
-- ==
-- entry: thesislike256
-- "Class 8192-256" script input { (mk_input 32i64 256i64) }

entry thesislike256 [m][d] (Q: [m][d][d]f16) (K: [m*d][d]f16) (V: [m*d][d]f16) =
  FlashAttention Q K V

--
-- ==
-- entry: thesislike512
-- "Class 8192-512" script input { (mk_input 16i64 512i64) }

entry thesislike512 [m][d] (Q: [m][d][d]f16) (K: [m*d][d]f16) (V: [m*d][d]f16) =
  FlashAttention Q K V