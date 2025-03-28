-- ==
-- entry: ludf16
-- compiled random input {[256][16][16]f16 [256][16][16]f16 [256][256][16][16]f16}
-- compiled random input {[256][32][32]f16 [256][32][32]f16 [256][256][32][32]f16}
-- compiled random input {[256][64][64]f16 [256][64][64]f16 [256][256][64][64]f16}
-- compiled random input {[256][128][128]f16 [256][128][128]f16 [256][256][128][128]f16}

-- Alternatively try this with default_tile_size=8 and default_reg_tile_size=2
-- compiled random input {[256][16][16]f16 [256][16][16]f16 [256][256][16][16]f16} auto output

-- ==
-- entry: ludf32
-- compiled random input {[256][16][16]f32 [256][16][16]f32 [256][256][16][16]f32}
-- compiled random input {[256][32][32]f32 [256][32][32]f32 [256][256][32][32]f32}
-- compiled random input {[256][64][64]f32 [256][64][64]f32 [256][256][64][64]f32}
-- compiled random input {[256][128][128]f32 [256][128][128]f32 [256][256][128][128]f32}

-- Alternatively try this with default_tile_size=8 and default_reg_tile_size=2
-- compiled random input {[256][16][16]f32 [256][16][16]f32 [256][256][16][16]f32} auto output

type real = f16

let ludMult [m][b] (top_per: [m][b][b]real, lft_per: [m][b][b]real, mat_slice: [m][m][b][b]real)
              : *[m][m][b][b]real =
  -- let top_slice = map transpose top_per in
  map (\(mat_arr: [m][b][b]real, lft: [b][b]real): [m][b][b]real  ->
        map (\ (mat_blk: [b][b]real, top: [b][b]real): [b][b]real  ->
                map  (\ (mat_row: [b]real, lft_row: [b]real): [b]real  ->
                        map2 (\mat_el top_row ->
                                let prods = map2 (*) lft_row top_row
                                let sum   = reduce (+) 0.0 prods
                                in mat_el - sum
                             ) mat_row (transpose top)
                    ) (zip (mat_blk) lft )
           ) (zip (mat_arr) (top_per) )
     ) (zip (mat_slice) (lft_per) )

def ludMultf16 [m][b] (top_per: [m][b][b]f16, lft_per: [m][b][b]f16, mat_slice: [m][m][b][b]f16)
              : *[m][m][b][b]f16 =
  -- let top_slice = map transpose top_per in
  map (\(mat_arr: [m][b][b]f16, lft: [b][b]f16): [m][b][b]f16  ->
        map (\ (mat_blk: [b][b]f16, top: [b][b]f16): [b][b]f16  ->
                map  (\ (mat_row: [b]f16, lft_row: [b]f16): [b]f16  ->
                        map2 (\mat_el top_row ->
                                let prods = map2 (*) lft_row top_row
                                let sum   = reduce (+) 0.0 prods
                                in mat_el - sum
                             ) mat_row (transpose top)
                    ) (zip (mat_blk) lft )
           ) (zip (mat_arr) (top_per) )
     ) (zip (mat_slice) (lft_per) )

def ludMultf32 [m][b] (top_per: [m][b][b]f32, lft_per: [m][b][b]f32, mat_slice: [m][m][b][b]f32)
              : *[m][m][b][b]f32 =
  -- let top_slice = map transpose top_per in
  map (\(mat_arr: [m][b][b]f32, lft: [b][b]f32): [m][b][b]f32  ->
        map (\ (mat_blk: [b][b]f32, top: [b][b]f32): [b][b]f32  ->
                map  (\ (mat_row: [b]f32, lft_row: [b]f32): [b]f32  ->
                        map2 (\mat_el top_row ->
                                let prods = map2 (*) lft_row top_row
                                let sum   = reduce (+) 0.0 prods
                                in mat_el - sum
                             ) mat_row (transpose top)
                    ) (zip (mat_blk) lft )
           ) (zip (mat_arr) (top_per) )
     ) (zip (mat_slice) (lft_per) )

let main [m][b] (top_per: [m][b][b]real)
                (lft_per: [m][b][b]real)
                (mat_slice: [m][m][b][b]real ) =
  ludMult (top_per, lft_per, mat_slice)
                -- (res_adj: [m][m][b][b]f32) =
    -- vjp ludMult (top_per, lft_per, mat_slice) res_adj

entry ludf16 [m][b] (top_per: [m][b][b]f16)
                (lft_per: [m][b][b]f16)
                (mat_slice: [m][m][b][b]f16 ) =
  ludMultf16 (top_per, lft_per, mat_slice)

entry ludf32 [m][b] (top_per: [m][b][b]f32)
                (lft_per: [m][b][b]f32)
                (mat_slice: [m][m][b][b]f32 ) =
  ludMultf32 (top_per, lft_per, mat_slice)

--- BIG COMMENT -------
--- The straigh compilation yields something like:
---
--- segmap(thread; #groups=num_tblocks_10633; tblocksize=segmap_tblock_size_10632; virtualise)
---     (gtid_9445 < m_9078, gtid_9446 < m_9078, gtid_9447 < b_9079, gtid_9448 < b_9079, gtid_9449 < b_9079) (~phys_tid_9450) :
---     { acc(acc_cert_p_10557, [m_9078][m_9078][b_9079][b_9079], {f32}),
---       acc(acc_cert_p_10590, [m_9078][m_9078][b_9079][b_9079], {f32})
---     } {
---
---         let {r_adj_el : f32} = r_adj[gtid_9445, gtid_9446, gtid_9447, gtid_9448]
---         let {lft_el : f32} = lft_per_9081[gtid_9445, gtid_9447, gtid_9449]
---         let {top_el : f32} = top_per_coalesced_10752[gtid_9446, gtid_9449, gtid_9448]
---
---         let {acc_10651 : acc(acc_cert_p_10590, [m_9078][m_9078][b_9079][b_9079], {f32})} =
---             update_acc(acc_p_10591, {gtid_9445, gtid_9446, gtid_9447, gtid_9449}, {r_adj_el * top_el})
---
---         let {acc_10652 : acc(acc_cert_p_10557, [m_9078][m_9078][b_9079][b_9079], {f32})} =
---             update_acc(acc_p_10558, {gtid_9445, gtid_9446, gtid_9448, gtid_9449}, {r_adj_el * lft_el})
---
---         return {returns acc_10652, returns acc_10651}
---     }
---
---     in {withacc_inter_10636, withacc_inter_10635})
---
---
--- segmap(thread; #groups=num_tblocks_10675; tblocksize=segmap_tblock_size_10674; virtualise)
---     (gtid_9334 < m_9078, gtid_9335 < m_9078, gtid_9336 < b_9079, gtid_9337 < b_9079) (~phys_tid_9338) :
---     { acc(acc_cert_p_10532, [m_9078][b_9079][b_9079], {f32}) }
---     {
---         let {r_adj_el : f32} = r_adj[gtid_9334, gtid_9335, gtid_9336, gtid_9337]
---         let {acc_10683 : acc(acc_cert_p_10532, [m_9078][b_9079][b_9079], {f32})} =
---           update_acc(acc_p_10533, {gtid_9334, gtid_9336, gtid_9337}, {r_adj_el})
---         return {returns acc_10683}
---     }
--------------------------------------------------
