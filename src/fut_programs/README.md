futmma autotune --backend=cuda --pass-option=-default-tile-size=16 --pass-option=-default-reg-tile-size=4 ./lud-cfal-orig.fut

futmma bench --backend=cudatc --pass-option=--nvrtc-option=-I../../../cutlass/include --tuning=xxx ./custom_attention_like.fut

./custom_attention_like --print-params

futmma cuda ./flash-full/flash-cfal-orig.fut && echo "$(<data/[16][16][16]f32.in)" "$(<data/[256][16]f32.in)" "$(<data/[256][16]f32.in)" | ./flash-full/flash-cfal-orig --entry-point=thesislike

futmma cuda ./flash-full/flash-cfal-orig.fut && echo "$(<data/[128][64][64]f32.in)" "$(<data/[8192][64]f32.in)" "$(<data/[8192][64]f32.in)"| ./flash-full/flash-cfal-orig --entry-point=thesislike