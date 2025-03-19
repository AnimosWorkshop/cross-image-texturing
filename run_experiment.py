import sys
import os
from datetime import datetime

if __name__ == "__main__":
    name = sys.argv[1]
    script_path = f"experiments/{name}/{name}_CIT.slurm"

    script_content = f"""#! /bin/sh
#SBATCH --job-name={name}_CIT
#SBATCH --output=logs/{name}_at_{datetime.now().strftime('%H:%M:%S')}/{name}_CIT.out
#SBATCH --error=logs/{name}_at_{datetime.now().strftime('%H:%M:%S')}/{name}_CIT.err
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

    # Ensure the experiment directory exists
    os.makedirs(f"experiments/{name}", exist_ok=True)

    # Write the script to a file
    with open(script_path, "w") as script_file:
        script_file.write(script_content)

    # Submit the job using sbatch
    os.system(f"sbatch {script_path}")

    # print(f"SLURM script written to {script_path}")
    print (script_content)
    print("\n\nThe experiment has been submitted to the cluster.")

    os.remove(script_path)
    print(f"SLURM script removed from {script_path}")