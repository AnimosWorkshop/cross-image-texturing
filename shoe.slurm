#! /bin/sh

#SBATCH --job-name=shoe_bg
#SBATCH --output=shoe_bg.out
#SBATCH --error=shoe_bg.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --time=10:00:00
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=24G

# echo "Reading config from $SLURM_SUBMIT_DIR/data/config.yaml"
python cit_run.py --config shoe/config.yaml