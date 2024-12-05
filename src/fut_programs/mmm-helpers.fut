

def matmulf32 [d] (A: [d][d]f16) (B: [d][d]f16) : [d][d]f32 =
  map (\Arow ->
         map (\Bcol ->
                map2 (*) Arow Bcol
                |> map f32.f16
                |> reduce (+) 0.0
             ) (transpose B)
      ) A

def matmulf16 [d] (A: [d][d]f16) (B: [d][d]f16) : [d][d]f16 =
  map (\Arow ->
         map (\Bcol ->
                map2 (*) Arow Bcol
                |> reduce (+) 0.0
             ) (transpose B)
      ) A
