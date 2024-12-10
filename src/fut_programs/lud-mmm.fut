-- ==
-- entry: lud128
-- compiled random input {[128][128][128]f16 [128][128][128]f16 [128][128][128][128]f32}

-- ==
-- entry: lud64
-- compiled random input {[128][64][64]f16 [128][64][64]f16 [128][128][64][64]f32}

import "mmm-helpers"
import "seq_acc"
     

let ludMult [m][b] (r: i64) (top_per: [m][b][b]f16, lft_per: [m][b][b]f16, mat_slice: [m][m][b][b]f32) =
  #[incremental_flattening(only_inner)]
  map (\(mat_arr: [m][b][b]f32, lft: [b][b]f16)  ->
         #[incremental_flattening(only_intra)]
         map (\(mat_blk: [b][b]f32, top: [b][b]f16)  ->
                let mm = matmulf32 lft top
                in seq_acc r (-) (copy mat_blk) (copy mm)
             ) (zip mat_arr top_per)
      ) (zip mat_slice lft_per)
  
-- entry lud64 [m] (top_per: [m][64][64]f16)
--          (lft_per: [m][64][64]f16)
--          (mat_slice: [m][m][64][64]f32 )
--           =
--   ludMult 32 (top_per, lft_per, mat_slice)

-- entry lud128 [m] (top_per: [m][128][128]f16)
--          (lft_per: [m][128][128]f16)
--          (mat_slice: [m][m][128][128]f32 )
--           =
--   ludMult 32 (top_per, lft_per, mat_slice)          

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
