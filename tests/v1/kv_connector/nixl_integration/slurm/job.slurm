#!/bin/bash
#SBATCH --job-name="xpyd benchmark"
#SBATCH --output="./logs/xpyd-benchmark-%j.out"
#SBATCH --error="./logs/xpyd-benchmark-%j.err"
#SBATCH --partition=gpuA40x4,gpuA100x4,gpuA100x4-interactive
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bcjw-delta-gpu
#SBATCH -t 00:01:00

# module reset
# module load cuda/12.4

source $nvme_dir/uv/vllm/bin/activate
export LD_LIBRARY_PATH=/sw/external/python/anaconda3/lib:$LD_LIBRARY_PATH
export UCX_NET_DEVICES=lo
export PATH=/u/kramesh/ucx/src/tools/info:$PATH

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

echo $nvme_dir

echo $UV_CACHE_DIR
