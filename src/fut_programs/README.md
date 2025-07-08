futmma autotune --backend=cuda --pass-option=-default-tile-size=16 --pass-option=-default-reg-tile-size=4 ./lud-cfal-orig.fut

futmma bench --backend=cudatc --pass-option=--nvrtc-option=-I../../cutlass/include --tuning=xxx ./custom_attention_like.fut

./custom_attention_like --print-params

futmma cuda ./flash-full/flash-cfal-orig.fut && echo "$(<data/[16][16][16]f32.in)" "$(<data/[256][16]f32.in)" "$(<data/[256][16]f32.in)" | ./flash-full/flash-cfal-orig --entry-point=thesislike

futmma cuda ./flash-full/flash-cfal-orig.fut && echo "$(<data/[128][64][64]f32.in)" "$(<data/[8192][64]f32.in)" "$(<data/[8192][64]f32.in)"| ./flash-full/flash-cfal-orig --entry-point=thesislike

futmma cudatc ./flash-full/flash-cfal-modified.fut

./flash-full/flash-cfal-modified --dump-cuda ./flash-full/kernel.cu

./flash-full/flash-cfal-modified --load-cuda ./flash-full/kernel.cu --nvrtc-option -I../../cutlass/include --entry-point thesislike32 < ./data/c_128-32-32f16_4096-32f16-4096-32f16.in > /dev/null

./flash-full/flash-cfal-modified --load-cuda ./flash-full/kernel-modified.cu --nvrtc-option -I../../cutlass/include --entry-point thesislike32 < ./data/c_128-32-32f16_4096-32f16-4096-32f16.in > /dev/null

futmma cudatc ./flash-cfal-modified.fut
nvcc -g -G -std=c++17 flash-cfal-modified.c -lcublas -I. -I/home/xzb272/Git/mma/cutlass/include/ -I../../../cutlass/tools/util/include -lOpenCL -lm -lcuda -lcudart -lnvrtc -o hand_compiled
cuda-gdb ./flash-full/hand_compiled
break 8433 / 8742
r --nvrtc-option=-I../../../cutlass/include --entry-point=thesislike16 <./data/c_128-16-16f16_2048-16f16_2048-16f16.in