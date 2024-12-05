

def matmulf32 [m][n][k] (A: [m][k]f16) (B: [k][n]f16) : [m][n]f32 =
  map (\Arow ->
         map (\Bcol ->
                map2 (*) Arow Bcol
                |> map f32.f16
                |> reduce (+) 0.0
             ) (transpose B)
      ) A

def matmulf16 [m][n][k] (A: [m][k]f16) (B: [k][n]f16) : [m][n]f16 =
  map (\Arow ->
         map (\Bcol ->
                map2 (*) Arow Bcol
                |> reduce (+) 0.0
             ) (transpose B)
      ) A
