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
  imap a (\a_row -> 
      imap b (\b_row -> 
          map2 (*) a_row b_row 
              |> map f32.f16
              |> reduce (+) 0f32
              |> f16.f32 
          ) 
      )

def matmul_intra [m][n][k] (a: [m][k]f16) (b: [k][n]f16) : [m][n]f16 =
  #[incremental_flattening(only_intra)]matmulT a (transpose b)
  --matmulT a (transpose b)

def unflatten_to 't [k] (n: i64) (m: i64) (A: [k]t) : [n][m]t =
  unflatten (A :> [n * m]t)

def combine [d] (m:i64) (A:[m][d][d]f16): [d][m*d]f16 =
    transpose (flatten A)

def add_rows [n][m] (pn:i64) (a:[n][m]f16): [n+pn][m]f16=
    let extra_rows = replicate pn (replicate m 0f16) --[pn][m]f16
    in (concat a extra_rows)

def add_cols [n][m] (pm:i64) (a:[n][m]f16): [n][m+pm]f16=
    let padded_rows = #[incremental_flattening(only_intra)]map (\row -> concat row (replicate pm 0f16)) a --[n][m+pm]f16
    in (padded_rows)

def add_both [n][m] (pn:i64) (pm:i64) (a:[n][m]f16): [n+pn][m+pm]f16=
    let padded_rows = add_cols pm a --[n][m+pm]f16
    let padded_cols = add_rows pn padded_rows--[n+pn][m+pm]f16
    in (padded_cols)


-- IS THIS NEEDED?
def matmulf32 [m][n][k] (A: [m][k]f16) (B: [k][n]f16) : [m][n]f16 =
  map (\Arow ->
         map (\Bcol ->
                map2 (*) Arow Bcol
                |> reduce (+) 0.0
             ) (transpose B)
      ) A

def matmul_tile_rows [d] (m: i64) (Qi:[d][d]f16) (K:[m*d][d]f16): [d][m*d]f16 =
  let split_k = unflatten_to m d K --: [m][d][d]f16
  let replicated_Qi = replicate m Qi --: [m][d][d]f16
  --let partials = #[incremental_flattening(only_intra)](map (matmulf32 (Qi)) split_k) --: [m][d][d]f16
  let partials = #[incremental_flattening(only_intra)]map2 matmulf32 replicated_Qi split_k --: [m][d][d]f16
  in (combine m partials) --[d][m*d]f16
  -- BOTH BELOW WORK
  --combine m (map (matmulT (Qi)) (unflatten_to m d K)) --[d][m*d]f16
  --combine m (#[incremental_flattening(only_intra)]map (matmulT (Qi)) (unflatten_to m d K)) --[d][m*d]f16

def tile 't [m][n] (M:i64) (N:i64) (a: [m][n]t): [M][N][m/M][n/N]t=
    map (\i ->
        map (\j ->
            map (\k ->
                map (\l -> a[i * (m/M) + k][j * (n/N) + l]) (iota (n/N))
            ) (iota (m/M))
        ) (iota (N))
  ) (iota (M))

def untile 't [M][N][m][n] (a:i64) (b:i64) (A: [M][N][m][n]t): [a][b]t =
    tabulate_2d (a) (b) (\i j ->
        A[i / m][0][j / n][j % n])

def matmul_tile_k [md][d] (A: [d][md]f16) (B: [md][d]f16) : [d][d]f16 =
    let m = md /d

    let split_a = map (\i -> 
        map (\j ->
            map (\k ->
                A[j][k+i*d]
            ) (iota d)
        ) (iota d)
    ) (iota m) -- [m][d][d]f16

    let split_b = unflatten_to m d B -- [m][d][d]f16

    let partials = map2 matmul_intra split_a split_b
    let result =  reduce (map2 (map2 (+))) (replicate d (replicate d 0.0f16)) partials
    in result

def dotproduct x y =
  map2 (*) x y
  |> reduce (+) 0f16

def matmul_tiling [m][n][k] (A: [m][k]f16) (B: [k][n]f16) =
    map (\ Arow ->
        map (\Bcol ->
            dotproduct Arow Bcol)
        (transpose B)
    ) A

let ne (m: i64) (n: i64) = (replicate (m * n) 0.0f16 |> unflatten)

def reduceOp [m][n] (acc: *[m][n]f16) (elm: [m][n]f16): [m][n]f16 =
  loop acc': *[m][n]f16 = (acc : *[m][n]f16) for i < m do
        acc' with [i, :] = map2 (+) elm[i] acc'[i]

def handleKBlocks [K][m][n][k] (Arow: [K][m][k]f16) (Bcol: [K][k][n]f16) : [m][n]f16 =
    let acc_init = ne m n in
    loop (acc: *[m][n]f16) = acc_init for K_i < K do
        let C = matmul_tiling Arow[K_i] Bcol[K_i]
        in reduceOp acc C

def run [M][K][N][m][n][k] (A: [M][K][m][k]f16) (B: [K][N][k][n]f16) : [M][N][m][n]f16 =
    #[incremental_flattening(only_inner)]map (\Arow ->
    --map (\Arow ->
        #[incremental_flattening(only_intra)]map (\Bcol ->
        --map (\Bcol ->
            handleKBlocks Arow Bcol
        ) (transpose B)
    ) A

def matmul_tile_mkn [md][d] (A: [d][md]f16) (B: [md][d]f16) : [d][d]f16 =
    -- TOOD more dynamic?
    let M = 8
    let K = 128
    let N = 8
    --let Atiles = tile M K A -- : [M][K][d/M][md/K] 
    --let Btiles = tile K N B -- : [K][N][md/K][d/N]
    --let C = run Atiles Btiles -- : [M][N][d/M][d/N]
    --let result = untile d d C -- : [d][d]
    --in result
    in untile d d (run (tile M K A) (tile K N B))

def oneIter [d] (m: i64) (K: [m*d][d]f16) (V: [m*d][d]f16) (Qi: [d][d]f16) : [d][d]f16 =
  -- If operating on too small a d dimension, pad it up to 64
  let P_block = if d<64i64 then (
    let n = 128i64-d

    -- Pad K
    let paddedK_bad = add_both (n*m) (n) K -- : [m*d+n*m][d+n]f16
    let paddedK_flat = flatten paddedK_bad -- : [(m*d+n*m)*(d+n)]f16
    let paddedK_good = unflatten_to (m*(d+n)) (d+n) paddedK_flat -- : [m*(d+n)][d+n]f16

    -- Pad Q
    let paddedQi = add_both (n) (n) Qi -- : [d+n][d+n]f16

    -- Get initial P_block
    --let padded_P_block = #[incremental_flattening(only_intra)]matmul_tile_rows m paddedQi paddedK_good |> opaque -- : [d][m*d]f16
    let padded_P_block = matmul_tile_rows m paddedQi paddedK_good |> opaque -- : [d][m*d]f16

    -- Unpad
    in (map (\r -> map (\l -> padded_P_block[r][l]) (iota (m*d))) (iota (d))) -- : [d][m*d]f16
  )
  -- If we don't need pading we can jump straight to the first matmul
  else (
    matmul_tile_rows m Qi K |> opaque -- : [d][m*d]f16
  )
  -- Always do the same softmax calculation
  let P_block = softmaxOnline P_block  -- : [d][m*d]f16
  
  in if d<64i64 -- small matrix, skip tensor cores
    then (matmulT P_block (transpose V)) -- : [d][d]f16
  else if (d>128) -- large matrix
    then (matmul_tile_mkn P_block V) -- : [d][d]f16
  else 
    (matmul_tile_k P_block V) -- : [d][d]f16

def oneIterSmall [d] (m: i64) (K: [m*d][d]f16) (V: [m*d][d]f16) (Qi: [d][d]f16) : [d][d]f16 =
  -- If operating on too small a d dimension, pad it up to 64
  let n = 128i64-d

  -- Pad K
  let paddedK_bad = add_both (n*m) (n) K -- : [m*d+n*m][d+n]f16
  let paddedK_flat = flatten paddedK_bad -- : [(m*d+n*m)*(d+n)]f16
  let paddedK_good = unflatten_to (m*(d+n)) (d+n) paddedK_flat -- : [m*(d+n)][d+n]f16

  -- Pad Q
  let paddedQi = add_both (n) (n) Qi -- : [d+n][d+n]f16

  -- Get initial P_block
  --let padded_P_block = #[incremental_flattening(only_intra)]matmul_tile_rows m paddedQi paddedK_good |> opaque -- : [d][m*d]f16
  let padded_P_block = matmul_tile_rows m paddedQi paddedK_good |> opaque -- : [d][m*d]f16

  -- Unpad
  let P_block = (map (\r -> map (\l -> padded_P_block[r][l]) (iota (m*d))) (iota (d))) -- : [d][m*d]f16

  -- Always do the same softmax calculation
  let P_block = softmaxOnline P_block  -- : [d][m*d]f16
  
  in (matmulT P_block (transpose V)) -- : [d][d]f16

def oneIterMid [d] (m: i64) (K: [m*d][d]f16) (V: [m*d][d]f16) (Qi: [d][d]f16) : [d][d]f16 =
  -- If we don't need pading we can jump straight to the first matmul
  let P_block = matmul_tile_rows m Qi K |> opaque -- : [d][m*d]f16

  -- Always do the same softmax calculation
  let P_block = softmaxOnline P_block  -- : [d][m*d]f16
  
  in (matmul_tile_k P_block V) -- : [d][d]f16

def oneIterMid_PRETILED [d] (m: i64) (K: [m][d][d]f16) (V: [m][d][d]f16) (Qi: [m][d][d]f16) : [m][d][d]f16 =
  let P_block = #[incremental_flattening(only_intra)]map2 matmulf32 Qi K --:[m][d][d]

  -- Always do the same softmax calculation
  --let P_block = softmaxOnline P_block  -- : [d][m*d]f16
  
  in (#[incremental_flattening(only_intra)]map2 matmulf32 P_block V) -- : [m][d][d]f16

def oneIterLarge [d] (m: i64) (K: [m*d][d]f16) (V: [m*d][d]f16) (Qi: [d][d]f16) : [d][d]f16 =
  let P_block = matmul_tile_rows m Qi K |> opaque -- : [d][m*d]f16

  -- Always do the same softmax calculation
  let P_block = softmaxOnline P_block  -- : [d][m*d]f16
  
  in (matmul_tile_mkn P_block V) -- : [d][d]f16

def FlashAttentionSmall [d][m] 
        (Q: [m][d][d]f16) 
        (K: [m*d][d]f16) 
        (V: [m*d][d]f16) 
      : [m][d][d]f16 =
  map (oneIterSmall m K V) Q

def FlashAttentionMid [d][m] 
        (Q: [m][d][d]f16) 
        (K: [m*d][d]f16) 
        (V: [m*d][d]f16) 
      : [m][d][d]f16 =
  map (oneIterMid m K V) Q

def FlashAttentionMid_PRETILED [d][m] 
        (Q: [m][m][d][d]f16) 
        (K: [m][d][d]f16) 
        (V: [m][d][d]f16) 
      : [m][m][d][d]f16 =
  map (oneIterMid_PRETILED m K V) Q

def FlashAttentionLarge [d][m] 
        (Q: [m][d][d]f16) 
        (K: [m*d][d]f16) 
        (V: [m*d][d]f16) 
      : [m][d][d]f16 =
  map (oneIterLarge m K V) Q

entry mk_input (m:i64) (d:i64): ([m][d][d]f16, [m*d][d]f16, [m*d][d]f16) =
  let Q = replicate d 1.0 |> replicate d |> replicate m
  let K = replicate d 1.0 |> replicate (m*d)
  let V = replicate d 1.0 |> replicate (m*d)
  in  (Q, K, V)

entry mk_input_PRETILED (m:i64) (d:i64): ([m][m][d][d]f16, [m][d][d]f16, [m][d][d]f16) =
  let Q = replicate d 1.0 |> replicate d |> replicate m |> replicate m
  let K = replicate d 1.0 |> replicate d |> replicate m
  let V = replicate d 1.0 |> replicate d |> replicate m
  in  (Q, K, V)

--
-- ==
-- entry: thesislike16
-- "Class 128-16 " script input { (mk_input 128i64 16i64) }

entry thesislike16 [m][d] (Q: [m][d][d]f16) (K: [m*d][d]f16) (V: [m*d][d]f16) =
  --FlashAttentionSmall Q K V
  FlashAttentionMid Q K V

--
-- ==
-- entry: thesislike16_PRETILED
-- "Class 128-16 " script input { (mk_input_PRETILED 128i64 16i64) }

entry thesislike16_PRETILED [m][d] (Q: [m][m][d][d]f16) (K: [m][d][d]f16) (V: [m][d][d]f16) =
  --FlashAttentionSmall Q K V
  FlashAttentionMid_PRETILED Q K V

--
-- ==
-- entry: thesislike32
-- "Class 128-32 " script input { (mk_input 128i64 32i64) }

entry thesislike32 [m][d] (Q: [m][d][d]f16) (K: [m*d][d]f16) (V: [m*d][d]f16) =
  FlashAttentionSmall Q K V

--
-- ==
-- entry: thesislike64
-- "Class 128-64 " script input { (mk_input 128i64 64i64) }

entry thesislike64 [m][d] (Q: [m][d][d]f16) (K: [m*d][d]f16) (V: [m*d][d]f16) =
  FlashAttentionMid Q K V

--
-- ==
-- entry: thesislike128
-- "Class 128-128" script input { (mk_input 128i64 128i64) }

entry thesislike128 [m][d] (Q: [m][d][d]f16) (K: [m*d][d]f16) (V: [m*d][d]f16) =
  FlashAttentionMid Q K V

--
-- ==
-- entry: thesislike256
-- "Class 128-256" script input { (mk_input 128i64 256i64) }

entry thesislike256 [m][d] (Q: [m][d][d]f16) (K: [m*d][d]f16) (V: [m*d][d]f16) =
  FlashAttentionMid Q K V

--
-- ==
-- entry: thesislike512
-- "Class 128-512" script input { (mk_input 128i64 512i64) }

entry thesislike512 [m][d] (Q: [m][d][d]f16) (K: [m*d][d]f16) (V: [m*d][d]f16) =
  FlashAttentionLarge Q K V
