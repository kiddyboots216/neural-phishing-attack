#!/bin/bash

#SBATCH     --nodes=1               # node count
#SBATCH     --ntasks-per-node=1      # total number of tasks per node
#SBATCH     --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH     --mem=32G                # total memory per node (4 GB per cpu-core is default)
#SBATCH     --gres=gpu:1             # number of gpus per node
#SBATCH     --time=23:59:00          # total run time limit (HH:MM:SS)
##SBATCH    --partition=mig            # partition (queue)
#SBATCH    --constraint=gpu80         # constraint (e.g. gpu80)
#SBATCH     -o Report/%j.out            # STDOUT
##SBATCH     --mail-type=ALL          # send email on job start, end and fail
##SBATCH     --mail-user=ashwinee@princeton.edu      # email address to send to

# replace these as necessary
CONDA_PATH=/scratch/gpfs/$USER/envs/llm
module purge 
module load anaconda3/2022.10
conda activate $CONDA_PATH
ulimit -n 50000

export PYTHONUNBUFFERED=1

python main.py --model_size ${1} --num_poisons ${2} --poisoning_rate ${3} --attack_type ${4} --clean_iters ${5} --num_digits ${6} --attack_inference_type ${7} --secret_prompts "${8}" --poison_prompts "${9}" --revision ${10} --phase_2p5_iters ${11} --secret_threshold ${12} --num_runs ${13} --dataset ${14} --seed ${15} --phase_4_iters ${16} --num_secrets ${17} --user_path /scratch/gpfs/$USER