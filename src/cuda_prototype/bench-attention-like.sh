#!/usr/bin/env sh

##############
# bM=bN=bK=128, threads=128
# 167 TFLops
# ./compile.sh 2 2 1 2 2 8 2 2 -DATTENTION_LIKE -DNUM_STAGES=1 && ./main 1024 256 0
###############
# bM=bN=bK=64, threads=64
# 84 TFLOPS
#./compile.sh 2 2 2 1 1 4 1 2  -DATTENTION_LIKE -DNUM_STAGES=1 && ./main 1024 256 0

##############
# bM=bN=bK=32, threads=128
#./compile.sh 2 2 2 2 1 4 2 2 -DATTENTION_LIKE -DNUM_STAGES=1 && ./main 1024 256

##############
# bM=bN=bK=16, threads=128
#./compile.sh 2 2 2 2 1 4 2 2 -DATTENTION_LIKE -DNUM_STAGES=1 && ./main 1024 256 0
##############
