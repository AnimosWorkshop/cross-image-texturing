#! /bin/sh

#SBATCH --job-name=wo_cn
#SBATCH --output=wo_cn.out
#SBATCH --error=wo_cn.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --time=10:00:00
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=20G

# echo "Reading config from $SLURM_SUBMIT_DIR/data/config.yaml"
# python cit_run.py --config data/config_w_controlnet.yaml
python cit_run.py --config data/config.yaml
# python cit_run.py \
#     --config $SLURM_SUBMIT_DIR/data/config.yaml
#     #--config /home/ML_courses/03683533_2024/lidor_yael_snir/snir-tests/cross-image-texturing/data/config.yaml