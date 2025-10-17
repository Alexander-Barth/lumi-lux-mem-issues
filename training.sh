#!/bin/bash -l
#SBATCH --job-name=FM
#SBATCH --account=project_465001568
#SBATCH --output=slurm-%x-%j.out

# run
# sbatch  --partition=small-g --time=12:00:00   --mem-per-cpu=70G --cpus-per-task=1  --ntasks=1 --nodes=1 --gpus=1  training-sat-vel.sh test_unet_vel_fm.jl

module load Local-CSC julia/1.12.0 julia-amdgpu/1.1.3

echo SLURM_JOB_NAME:       $SLURM_JOB_NAME
echo SLURM_JOB_NODELIST:   $SLURM_JOB_NODELIST
echo ROCR_VISIBLE_DEVICES: $ROCR_VISIBLE_DEVICES
echo julia:                $(which julia)
echo arguments:            $@

export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache-$USER-$$"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
rm -rf "$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"


export JULIA_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export JULIA_HISTORY="$HOME/.julia/logs/repl_history.jl"
if [ -e Project.toml ]; then
    export JULIA_PROJECT="$PWD"
fi

# https://docs.lumi-supercomputer.eu/development/compiling/prgenv/#gpu-aware-mpi
export MPICH_GPU_SUPPORT_ENABLED=1
# on each node but only once per node

if [ "$SLURM_NTASKS" -gt "1" ]; then
    export PARALLEL=true
else
    export PARALLEL=false
fi

srun julia "$@"
