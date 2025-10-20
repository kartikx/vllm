#!/bin/bash
#SBATCH --job-name="xpyd benchmark rps_15"
#SBATCH --output="./slurm/xpyd-benchmark-rps_15.out"
#SBATCH --error="./slurm/xpyd-benchmark-rps_15.err"
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

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

source $nvme_dir/uv/vllm/bin/activate
export LD_LIBRARY_PATH=/sw/external/python/anaconda3/lib:$LD_LIBRARY_PATH
export UCX_NET_DEVICES=lo

export MODELS="meta-llama/Llama-3.1-8B"
export PREFILLER_TP_SIZE=1
export DECODER_TP_SIZE=1
export RESULT_DIR="./results"
export PD_RATIO_LIST="1,1 2,1 3,1 2,2 1,3"
export INPUT_LENS="1000"
export OUTPUT_LENS="10"
export NUM_PROMPTS="100"
export RPS="20"

srun bash ./benchmark_xpyd.sh
