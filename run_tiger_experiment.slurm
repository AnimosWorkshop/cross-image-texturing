#! /bin/sh

#SBATCH --job-name=tiger_CIT
#SBATCH --output=tiger_CIT.out
#SBATCH --error=tiger_CIT.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --time=01:00:00
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=24G

python cit_run.py --config animal/config.yaml
