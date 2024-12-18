#!/bin/bash --login

#SBATCH -D ./                           # Working Directory (Can stay as it is)
#SBATCH -J PINNuv_55k            # Job Name (Displayed in squeue --me command)
#SBATCH -o ./Output/PINN.%j.%N.out  # Output-File (Saved in Output folder with the given name, this file will print everything that you usually print in console)
#SBATCH --ntasks=4                      # Anzahl Prozesse P (CPU-Cores, can stay as it is)
#SBATCH --cpus-per-task=1               # Anzahl CPU-Cores pro Prozess P (Can stay as it is)
#SBATCH --gres=gpu:1,gpu_mem:10G,ccc:75 # 1 GPU required with 10GB memory and compute capability of 7.5 (Can stay as it is)
#SBATCH --mem=20G                       # Change memory if needed (in GB, 10G is good enough for now)
#SBATCH --time=2000                     # Time to run (minutes)

#Auf GPU-Knoten rechnen:
#Job-Status per Mail:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=donner@campus.tu-berlin.de   # Email address (Change to your email address)

module load cuda/12.0
conda activate /work/yadav/tf-env/         # Load the environment (I guess this environment is fine)   

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA
deviceQuery

python3 main_cyl_c.py # Run your python script (file name)
