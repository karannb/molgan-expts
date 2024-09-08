#!/bin/sh -x

#SBATCH --job-name=temp
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --output=logs/out_%j.log
#SBATCH --error=logs/err_%j.log
#SBATCH --time=10:00:00

python3 trainer.py \
    --name $1 \
    --mode $2 \
    --dataset $3 \
    --dropout $4 \
    --generator_step 0.2