#!/bin/sh
#SBATCH --job-name=cgarnier_reconstruction
#SBATCH --account=ad:ard-online
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --gpus=1
#SBATCH --time=01:00:00


export CONDA_PREFIX=/sdf/group/ad/beamphysics/rroussel/miniforge3/
export PATH=${CONDA_PREFIX}/bin/:$PATH
source ${CONDA_PREFIX}/etc/profile.d/conda.sh
conda activate gpsr

python run_reconstruction.py \
  --processed_data /sdf/data/ad/ard-online/GPSR/20250719/processed_data/6d_data_1752998168.dset \
  --dump_location /sdf/home/c/cgarnier/machine_learning_setup/results/ \
  --diag0_lattice_file $LCLS_LATTICE/cheetah/diag0.json  \
  --hyper_params hyper_params.yaml