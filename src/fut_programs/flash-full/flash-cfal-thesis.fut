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

def softmaxChunkML16 (q: i64) (xs_glb: [q*chunk]f16) : (f16,f16) = #[unsafe]
  let g = chunk/fseq in
  loop (mi_old : f16, li_old : f16) = (f16.lowest, 0.0)
  for i < q do
    let xs = copy2shr ( xs_glb[i*chunk: i*chunk + chunk] :> [g*fseq]f16 )
    let xs = xs
    --
    let maxi = reduceEffSeq id f16.max f16.lowest xs
    let sumi = reduceEffSeq (\x -> f16.exp (x - maxi)) (+) 0.0 xs
    --
    let mi_new = f16.max mi_old maxi
    let eij = f16.exp (maxi - mi_new)
    let eli = li_old * (f16.exp (mi_old - mi_new))
    let li_new = eli + sumi * eij
    in  (mi_new, li_new) 

def softmaxOnline [m][n] (xss: [m][n]f16) : [m][n]f16 = #[unsafe]
  let q = assert (n % chunk == 0) (n / chunk)
  let (mis, lis) = 
          #[incremental_flattening(only_intra)]
          map (softmaxChunkML16 q) (xss :> [m][q*chunk]f16)
      |> unzip |> opaque
  in  imap3 xss mis lis
        (\ xs mi li ->
          map (\ x -> f16.exp (x - mi) / li ) xs
        ) |> opaque

def matmulT [m][n][k] (a: [m][k]f16) (b: [n][k]f16) : [m][n]f16 =
  imap a (\a_row -> imap b (\b_row -> map2 (*) a_row b_row |> reduce (+) 0.0) )

def matmul [m][n][k] (a: [m][k]f16) (b: [k][n]f16) : [m][n]f16 =
  matmulT a (transpose b)

def matmulThesis [m][n][k] (A: [m][k]f16) (B: [k][n]f16) : [m][n]f16 =
  map (\Arow ->
    map (\Bcol ->
      map2 (*) Arow Bcol
        |> reduce (+) 0.0
    ) (transpose B)
  ) A

def mm [d] (A: [d][d]f16) (B: [d][d]f16) : [d][d]f16 =
  map (\Arow ->
         map (\Bcol ->
                map2 (*) Arow Bcol
                |> reduce (+) 0.0
             ) (transpose B)
      ) A

def oneIter [d] (K: [d][d]f16) (V: [d][d]f16) (Qi: [d][d]f16) : [d][d]f16 =
  let P_block = mm Qi K --|> opaque -- : [d][m*d]f16 
  --let P_block = softmaxOnline P_block
  in mm P_block V      -- : [d][d]f16

def FlashAttention [d][m] 
        (Q: [m][d][d]f16) 
        (K: [d][d]f16) 
        (V: [d][d]f16) 
      : [m][d][d]f16 =
  map (oneIter K V) Q

entry mk_input (m:i64) (d:i64) : ([m][d][d]f16, [d][d]f16, [d][d]f16) =
  let Q = replicate d 1.0 |> replicate d |> replicate m
  let K = replicate d 1.0 |> replicate d
  let V = replicate d 1.0 |> replicate d
  in (Q, K, V)

--
-- ==
-- entry: thesislike16
-- "Block 8192-16 " only_intra script input { (mk_input 100000 16i64) }
entry thesislike16 [m] (Q: [m][16][16]f16) (K: [16][16]f16) (V: [16][16]f16) =
  #[incremental_flattening(only_intra)]FlashAttention Q K V

--
-- ==
-- entry: thesislike32
-- "Block 8192-32 " only_intra script input { (mk_input 100000 32i64) }
entry thesislike32 [m] (Q: [m][32][32]f16) (K: [32][32]f16) (V: [32][32]f16) =
  #[incremental_flattening(only_intra)]FlashAttention Q K V

--
-- ==
-- entry: thesislike64
-- "Block 8192-64 " only_intra script input { (mk_input 100000 64i64) }
entry thesislike64 [m] (Q: [m][64][64]f16) (K: [64][64]f16) (V: [64][64]f16) =
  #[incremental_flattening(only_intra)]FlashAttention Q K V

--
-- ==
-- entry: thesislike128
-- "Block 8192-128" only_intra script input { (mk_input 100000 128i64) }
entry thesislike128 [m] (Q: [m][128][128]f16) (K: [128][128]f16) (V: [128][128]f16) =
  #[incremental_flattening(only_intra)]FlashAttention Q K V

----
---- ==
---- entry: thesislike256
---- "Block 8192-256" only_intra script input { (mk_input 100000 256i64) }
--entry thesislike256 [m] (Q: [m][256][256]f16) (K: [256][256]f16) (V: [256][256]f16) =
--  #[incremental_flattening(only_intra)]FlashAttention Q K V
--
----
---- ==
---- entry: thesislike512
---- "Class 8192-512" only_intra script input { (mk_input 100000 512i64) }
--entry thesislike512 [m] (Q: [m][512][512]f16) (K: [512][512]f16) (V: [512][512]f16) =
--  #[incremental_flattening(only_intra)]FlashAttention Q K V

