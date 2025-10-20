#!/bin/bash
#SBATCH --job-name="xpyd benchmark-8b ss"
#SBATCH --output="./logs/xpyd-benchmark-8b-ss.out"
#SBATCH --error="./logs/xpyd-benchmark-8b-ss.err"
#SBATCH --partition=gpuA100x4,gpuA100x4-interactive
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest
#SBATCH --account=bcjw-delta-gpu
#SBATCH -t 00:60:00

# module reset
# module load cuda/12.4

echo -e "job $SLURM_JOBID is starting on `hostname`at $(date)\n\n"

source $nvme_dir/uv/vllm/bin/activate
export LD_LIBRARY_PATH=/sw/external/python/anaconda3/lib:$LD_LIBRARY_PATH
export UCX_NET_DEVICES=lo

export MODELS="meta-llama/Llama-3.1-8B"
export RESULT_DIR="./results/8B_ss"
export PD_RATIO_LIST="1,1 2,1 3,1"
export INPUT_LENS="1000"
export OUTPUT_LENS="100"
export NUM_PROMPTS="100"
export RPS="20"

srun bash ./benchmark_xpyd_shared_storage.sh