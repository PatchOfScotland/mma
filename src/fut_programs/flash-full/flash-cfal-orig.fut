def imap xs f = map f xs
def imap3 xs ys zs f = map3 f xs ys zs

def chunk : i64 = 1024
def fseq  : i64 = 32

def copy2shr16 [n] (xs: [n]f16) : *[n]f16 = #[unsafe]
  let xs' = copy xs
  in  if opaque(true) then xs'
      else xs' with [0] = 0f16

def copy2shr32 [n] (xs: [n]f32) : *[n]f32 = #[unsafe]
  let xs' = copy xs
  in  if opaque(true) then xs'
      else xs' with [0] = 0f32

def reduceEffSeq16 [g] (f: f16 -> f16) (bop: f16->f16->f16) (ne: f16) (xs: [g*fseq]f16) : f16 = #[unsafe]
  let redPerThd (tid: i64) =
    loop r = ne for i < fseq do
        bop r (f xs[i*g + tid])
  -- per-thread reduce
  let rs = map redPerThd (iota g)
  in  reduce_comm bop ne rs

def reduceEffSeq32 [g] (f: f32 -> f32) (bop: f32->f32->f32) (ne: f32) (xs: [g*fseq]f32) : f32 = #[unsafe]
  let redPerThd (tid: i64) =
    loop r = ne for i < fseq do
        bop r (f xs[i*g + tid])
  -- per-thread reduce
  let rs = map redPerThd (iota g)
  in  reduce_comm bop ne rs

def softmaxChunkML16 (q: i64) (xs_glb: [q*chunk]f16) : (f16,f16) = #[unsafe]
  let g = chunk/fseq in
  loop (mi_old : f16, li_old : f16) = (f16.lowest, 0.0)
  for i < q do
    let xs = copy2shr16 ( xs_glb[i*chunk: i*chunk + chunk] :> [g*fseq]f16 )
    let xs = xs
    --
    let maxi = reduceEffSeq16 id f16.max f16.lowest xs
    let sumi = reduceEffSeq16 (\x -> f16.exp (x - maxi)) (+) 0.0 xs
    --
    let mi_new = f16.max mi_old maxi
    let eij = f16.exp (maxi - mi_new)
    let eli = li_old * (f16.exp (mi_old - mi_new))
    let li_new = eli + sumi * eij
    in  (mi_new, li_new) 

def softmaxChunkML32 (q: i64) (xs_glb: [q*chunk]f32) : (f32,f32) = #[unsafe]
  let g = chunk/fseq in
  loop (mi_old : f32, li_old : f32) = (f32.lowest, 0.0)
  for i < q do
    let xs = copy2shr32 ( xs_glb[i*chunk: i*chunk + chunk] :> [g*fseq]f32 )
    let xs = xs
    --
    let maxi = reduceEffSeq32 id f32.max f32.lowest xs
    let sumi = reduceEffSeq32 (\x -> f32.exp (x - maxi)) (+) 0.0 xs
    --
    let mi_new = f32.max mi_old maxi
    let eij = f32.exp (maxi - mi_new)
    let eli = li_old * (f32.exp (mi_old - mi_new))
    let li_new = eli + sumi * eij
    in  (mi_new, li_new) 

def softmaxOnline16 [m][n] (xss: [m][n]f16) : [m][n]f16 = #[unsafe]
  let q = assert (n % chunk == 0) (n / chunk)
  let (mis, lis) = 
          #[incremental_flattening(only_intra)]
          map (softmaxChunkML16 q) (xss :> [m][q*chunk]f16)
      |> unzip |> opaque
  in  imap3 xss mis lis
        (\ xs mi li ->
          map (\ x -> f16.exp (x - mi) / li ) xs
        ) |> opaque

def softmaxOnline32 [m][n] (xss: [m][n]f32) : [m][n]f32 = #[unsafe]
  let q = assert (n % chunk == 0) (n / chunk)
  let (mis, lis) = 
          #[incremental_flattening(only_intra)]
          map (softmaxChunkML32 q) (xss :> [m][q*chunk]f32)
      |> unzip |> opaque
  in  imap3 xss mis lis
        (\ xs mi li ->
          map (\ x -> f32.exp (x - mi) / li ) xs
        ) |> opaque

def matmulT16 [m][n][k] (a: [m][k]f16) (b: [n][k]f16) : [m][n]f16 =
  imap a (\a_row -> imap b (\b_row -> map2 (*) a_row b_row |> reduce (+) 0.0) )

def matmulT32 [m][n][k] (a: [m][k]f32) (b: [n][k]f32) : [m][n]f32 =
  imap a (\a_row -> imap b (\b_row -> map2 (*) a_row b_row |> reduce (+) 0.0) )

def matmul16 [m][n][k] (a: [m][k]f16) (b: [k][n]f16) : [m][n]f16 =
  matmulT16 a (transpose b)

def matmul32 [m][n][k] (a: [m][k]f32) (b: [k][n]f32) : [m][n]f32 =
  matmulT32 a (transpose b)

def oneIter16 [d][m] (K: [m*d][d]f16) (V: [m*d][d]f16) (Qi: [d][d]f16) : [d][d]f16 =
  let P_block = matmulT16 Qi K |> opaque -- : [d][m*d]f16 
  -- let P_block = softmax P_block
  let P_block = softmaxOnline16 P_block -- : [d][m*d]f16 
  in  matmul16 P_block V      -- : [d][d]f16

def oneIter32 [d][m] (K: [m*d][d]f32) (V: [m*d][d]f32) (Qi: [d][d]f32) : [d][d]f32 =
  let P_block = matmulT32 Qi K |> opaque -- : [d][m*d]f32 
  -- let P_block = softmax P_block
  let P_block = softmaxOnline32 P_block -- : [d][m*d]f16 
  in  matmul32 P_block V      -- : [d][d]f32

def FlashAttention16 [d][m] 
        (Q: [m][d][d]f16) 
        (K: [m*d][d]f16) 
        (V: [m*d][d]f16) 
      : [m][d][d]f16 =
  map (oneIter16 K V) Q
  
def FlashAttention32 [d][m] 
        (Q: [m][d][d]f32) 
        (K: [m*d][d]f32) 
        (V: [m*d][d]f32) 
      : [m][d][d]f32 =
  map (oneIter32 K V) Q

entry mk_input16 (m:i64) (d:i64) : ([m][d][d]f16, [m*d][d]f16, [m*d][d]f16) =
  let Q = replicate d 1.0 |> replicate d |> replicate m
  let K = replicate d 1.0 |> replicate (m*d)
  let V = replicate d 1.0 |> replicate (m*d)
  in  (Q, K, V)
  
entry mk_input32 (m:i64) (d:i64) : ([m][d][d]f32, [m*d][d]f32, [m*d][d]f32) =
  let Q = replicate d 1.0 |> replicate d |> replicate m
  let K = replicate d 1.0 |> replicate (m*d)
  let V = replicate d 1.0 |> replicate (m*d)
  in  (Q, K, V)

--
-- ==
-- entry: thesislike16
-- "Class 128-16 " script input { (mk_input16 128i64 16i64) }
-- "Class 128-32 " script input { (mk_input16 128i64 32i64) }
-- "Class 128-64 " script input { (mk_input16 128i64 64i64) }
-- "Class 128-128" script input { (mk_input16 128i64 128i64) }
-- "Class 128-256" script input { (mk_input16 128i64 256i64) }
-- "Class 128-512" script input { (mk_input16 128i64 512i64) }
entry thesislike16 [m][d] (Q: [m][d][d]f16) (K: [m*d][d]f16) (V: [m*d][d]f16) =
  FlashAttention16 Q K V

--
-- ==
-- entry: thesislike32
-- "Class 128-16 " script input { (mk_input32 128i64 16i64) }
-- "Class 128-32 " script input { (mk_input32 128i64 32i64) }
-- "Class 128-64 " script input { (mk_input32 128i64 64i64) }
-- "Class 128-128" script input { (mk_input32 128i64 128i64) }
-- "Class 128-256" script input { (mk_input32 128i64 256i64) }
-- "Class 128-512" script input { (mk_input32 128i64 512i64) }
entry thesislike32 [m][d] (Q: [m][d][d]f32) (K: [m*d][d]f32) (V: [m*d][d]f32) =
  FlashAttention32 Q K V
