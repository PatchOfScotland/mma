# Futhark Compiler

## Overview
- src.Language.Futhark.Prop.hs extracts intrinsics
- 

# 24/9

- Use Cutlass basic blocks
- Call into them. Make them a header file

## Intragroup kernel focus

```futhark
let dotproduct [n] (x: [n]f16) (y: [n]f16) =
    #[sequential]map2 (*) x y |> reduce (+) 0

let matmul16 (A: [16][16]f16) (B: [16][16]f16) : [16][16]f16 =
    map (\ Arow -> map (\Bcol -> dotproduct Arow Bcol) (transpose B)) A

let intra_block_mmm [k] (A: [k][16][16]f16) (B: [k][16][16]f16) : [k][16][16]f16 =    
    map2 matmul16 A B
```

# Links

- https://www.cs.utexas.edu/~flame/BLISRetreat2023/slides/Thakkar_BLISRetreat2023.pdf
- https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9593-cutensor-high-performance-tensor-operations-in-cuda-v2.pdf
- https://www.nvidia.com/en-us/on-demand/session/gtcsj20-s21745/
- https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21745-developing-cuda-kernels-to-push-tensor-cores-to-the-absolute-limit-on-nvidia-a100.pdf


## Compiler modifications

### Option 1a: noninlined function calls within segmap(inblock)
+ May be possible to just translate function defs at codegen, so no need to modify impgen

- Need memory fixups, function args in global, avoid extra shared usage
- Currently not possible to pass arrays to functions as args in cuda backend
- No feasible way to return entire array as ouput, need to calculate placement in futhark
- Seemingly limited/no support for device functions in futhark

### Option 1b: noninlined function calls directly in segmap(block)
+ May be possible to just translate function defs at codegen, so no need to modify impgen
+ Can get whole array and return, full control

- Need memory fixups, function args in global, avoid extra shared usage
- Currently only compiles with typechecker off
- Seemingly limited/no support for device functions in futhark


### Option 2: segmap(inblock) with attribute, maybe function call inside
~ Pattern match in impgen, replace by new constructs, catch these at codegen

+ less memory fixups?

- Need to add new impcode constructs (KernelOps)
- May be harder to avoid interference from other passes?


### Option 3: gather and scatter, give indexes and values to futhark
+ Native support
+ Minimal memory fixup

- No async copies
- May be hard to avoid extra memory operations





## Codegen functions
Since inblock: assume global memory slice with known dimensions, that fit in shared


Have gemm output global memory, no registers needed, or add copy registers -> global function




## TODO:

Mål:
Generelt for størrelser, i første omgang med begrænsede muligheder, måske dynamiske værdier.
Generelt for typer, f16, f32, f64, mixed precision
Generelt for antal outer dims
Match seg(thread), måske reg-tiling output
Make fixup less agressive



Generelt hvordan?:
Template haskell vs template c++?
Static args hvordan
Instatier flere c++ templates, eventuelt med template Haskell
Dynamisk if til at vælge, eventuelt statisk ved at fange i codegen
Sæt shared size i første pass, giv størrelser som input
Se om variable størrelser giver problemer i compileren



Refactoring:
Match fra seg(block)?



Andet:
Benchmarks
Tests
Flash attention



Fix explicit allocations
