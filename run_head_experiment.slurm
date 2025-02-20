#! /bin/sh

#SBATCH --job-name=head_CIT
#SBATCH --output=head_CIT.out
#SBATCH --error=head_CIT.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --time=01:00:00
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=24G

python cit_run.py --config data/config.yaml # this one is the Kratos experiment (human head)
# python cit_run.py --config dragon/config.yaml
# python cit_run.py --config shoe/config.yaml
