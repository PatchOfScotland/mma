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