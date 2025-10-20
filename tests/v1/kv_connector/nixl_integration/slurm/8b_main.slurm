#!/bin/bash
#SBATCH --job-name="xpyd benchmark 8b"
#SBATCH --output="./logs/xpyd-benchmark-8b-main-%j.out"
#SBATCH --error="./logs/xpyd-benchmark-8b-main-%j.err"
#SBATCH --partition=gpuA100x4,gpuA100x4-interactive
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest
#SBATCH --account=bcjw-delta-gpu
#SBATCH -t 00:60:00

# module reset
module load cuda/12.4

# setup
source $nvme_dir/uv/vllm/bin/activate

echo "vllm with block size 1"

export UCX_HOME=$HOME/ucx
export UCX_TLS=cuda_ipc,cuda_copy,sm,self,tcp
export UCX_NET_DEVICES=lo

export LD_LIBRARY_PATH=$UCX_HOME/lib:/sw/external/python/anaconda3/lib:$LD_LIBRARY_PATH
export PATH=$UCX_HOME/bin:$PATH
export UCX_PROTO_INFO=y
# export UCX_LOG_LEVEL=debug
export PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc_dir
mkdir -p $PROMETHEUS_MULTIPROC_DIR

srun nvidia-smi topo -m

export MODELS="meta-llama/Llama-3.1-8B"
export RESULT_DIR="./results/8B"
export PD_RATIO_LIST="1,1 2,1 3,1 1,2 1,3 2,2"
export INPUT_OUTPUT_LIST="1000,100"
# Allow overriding NUM_PROMPTS and RPS via environment variables when submitting
# e.g. `sbatch --export=ALL,NUM_PROMPTS=1000,RPS=50 8b_main.slurm`
# If not provided, fall back to the previous defaults
export NUM_PROMPTS_LIST="${NUM_PROMPTS:-500}"
export RPS_LIST="${RPS:-20}"

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

echo "NUM_PROMPTS=$NUM_PROMPTS"
echo "RPS=$RPS"

srun bash ./benchmark_xpyd.sh
