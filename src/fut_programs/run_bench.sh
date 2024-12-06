#!/usr/bin/env sh

echo "Starting autotune..."
echo "--------------------"
futmma autotune --backend=cuda attention_like_orig.fut
futmma autotune --backend=cuda custom_attention_like_orig.fut
futmma autotune --backend=cuda batched_mmm_orig.fut
echo "--------------------"
echo "Autotune done..."
echo "Starting bench..."
echo "--------------------"
futmma bench --backend=cuda attention_like_orig.fut
futmma bench --backend=cuda custom_attention_like_orig.fut
futmma bench --backend=cuda batched_mmm_orig.fut
echo "--------------------"
echo "Bench done..."
echo "Starting bench tensor cores..."
echo "--------------------"
futmma bench --backend=cudatc --pass-option=--nvrtc-option=-I../../cutlass/include attention_like.fut
futmma bench --backend=cudatc --pass-option=--nvrtc-option=-I../../cutlass/include custom_attention_like.fut
futmma bench --backend=cudatc --pass-option=--nvrtc-option=-I../../cutlass/include batched_mmm.fut
