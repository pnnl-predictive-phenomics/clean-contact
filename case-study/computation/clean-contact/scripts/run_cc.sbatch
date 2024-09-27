#!/bin/sh
#SBATCH -A ppi_concerto
#SBATCH -p a100_shared
#SBATCH -t 4-0
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -o run_cc_out.txt
#SBATCH -e run_cc_errors.txt

module purge
module load gcc/13.1.0
module load cuda/11.8
module load python/miniconda23.5.2

source /share/apps/python/miniconda23.5.2/etc/profile.d/conda.sh
conda activate clean-contact
export PATH=/people/jerg881/.conda/envs/clean-contact/bin:$PATH

cd /qfs/projects/ppi_concerto/Abby/CLEAN-Contact/CLEAN-Contact
echo "about to run 'python run_cc.py' from run_cc.sbatch"
python run_cc.py
echo "end of run_cc.sbatch"
