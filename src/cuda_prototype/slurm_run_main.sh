#!/bin/bash

module load cuda/12.2

hostname
nvidia-smi

make
./main
