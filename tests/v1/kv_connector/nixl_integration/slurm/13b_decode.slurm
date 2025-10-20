#!/bin/bash
#SBATCH --job-name="xpyd benchmark decode"
#SBATCH --output="./slurm/xpyd-benchmark-13b-decode.out"
#SBATCH --error="./slurm/xpyd-benchmark-13b-decode.err"
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

export MODELS="meta-llama/Llama-2-13b-hf"
export RESULT_DIR="./results/13B"
export PD_RATIO_LIST="3,1"
export INPUT_LENS="100"
export OUTPUT_LENS="100"
export NUM_PROMPTS="500"
export DECODE_MAX_NUM_BATCHED_TOKENS="256"
export RPS="inf"

echo "max num batched tokens = $DECODE_MAX_NUM_BATCHED_TOKENS"
srun bash ./benchmark_xpyd.sh

export DECODE_MAX_NUM_BATCHED_TOKENS="2048"
echo "max num batched tokens = $DECODE_MAX_NUM_BATCHED_TOKENS"
srun bash ./benchmark_xpyd.sh