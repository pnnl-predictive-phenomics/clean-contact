#!/bin/sh
#SBATCH -A ppi_concerto
#SBATCH -p dl_shared
#SBATCH -t 4-0
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -o run_clean_out.txt
#SBATCH -e run_clean_errors.txt

module purge
module load gcc/13.1.0
module load cuda/12.1
module load python/miniconda23.5.2

source /share/apps/python/miniconda23.5.2/etc/profile.d/conda.sh
conda activate clean
export PATH=/people/jerg881/.conda/envs/clean/bin:$PATH

cd /qfs/projects/ppi_concerto/Abby/original_CLEAN/CLEAN/app
echo "about to run 'python run_clean.py' from run_clean.sbatch"
python run_clean.py
echo "end of run_clean.sbatch"
