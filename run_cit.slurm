#! /bin/sh

#SBATCH --job-name=MergedMVD
#SBATCH --output=Merged.out
#SBATCH --error=Merged.err
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-research
#SBATCH --time=10:00:00
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=20G

python cit_run.py \
    --config /home/ML_courses/03683533_2024/lidor_yael_snir/cross-image-texturing/data/config.yaml