-- ==
-- entry: run_orig
-- compiled script input { (mk_input 100000 16) }
-- compiled script input { (mk_input 100000 32) }
-- compiled script input { (mk_input 100000 64) }
-- compiled script input { (mk_input 100000 128) }

import "custom_attention_like"

entry run_orig [m][d] (Q: [m][d][d]real) (K: [d][d]real) (V: [d][d]real) =
  flashAttention Q K V
