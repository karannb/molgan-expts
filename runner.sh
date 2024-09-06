#!/bin/sh -x

python3 trainer.py \
    --name some_name \
    --mode gumbel \
    --dataset qm7 \
    --dropout 0.1 \
    --generator_step 0.2