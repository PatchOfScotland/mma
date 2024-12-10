
def seq_acc [m][n] 'a
  (seq_factor: i64)
  (op: a->a->a)
  (acc: *[m][n]a)
  (C: *[m][n]a) =
  let thrd_work = m * n / seq_factor in
  loop acc' = acc for j < thrd_work do
  let js_per_row = n / seq_factor in
  let row = j / js_per_row
  let col = j % js_per_row
  let col_offset = col * seq_factor
  in acc' with [row, col_offset:col_offset + seq_factor] =
    tabulate seq_factor (\k -> op acc'[row, col_offset + k] C[row, col_offset + k])
