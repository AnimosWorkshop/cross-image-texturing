# construct and run a slurm script
import sys
import os
if __name__ == "__main__":
    name = sys.argv[1]
    
    command = f"""#! /bin/sh
    #SBATCH --job-name=experiments/{name}_CIT
    #SBATCH --output=experiments/{name}/{name}_CIT.out
    #SBATCH --error=experiments/{name}/{name}_CIT.err
    #SBATCH --partition=killable
    #SBATCH --account=gpu-research
    #SBATCH --time=01:00:00
    #SBATCH --signal=USR1@120
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --gpus=1
    #SBATCH --mem-per-gpu=24G

    python src/cit_run.py --config experiments/{name}/config.yaml
    """
    os.system(command)