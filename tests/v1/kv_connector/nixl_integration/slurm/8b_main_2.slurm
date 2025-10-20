#!/bin/bash
#SBATCH --job-name="xpyd benchmark 8b main"
#SBATCH --output="./logs/xpyd-benchmark-8b-main-%j.out"
#SBATCH --error="./logs/xpyd-benchmark-8b-main-%j.err"
#SBATCH --partition=gpuA100x4,gpuA100x4-interactive,gpuA100x8,gpuA100x8-interactive
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest
#SBATCH --account=bcjw-delta-gpu
#SBATCH -t 00:60:00

# module reset
module load cuda/12.4

# setup
source $nvme_dir/uv/vllm/bin/activate

export UCX_HOME=$HOME/ucx
export UCX_TLS=tcp,cuda_ipc,cuda_copy,sm,self
export UCX_NET_DEVICES=lo

export LD_LIBRARY_PATH=$UCX_HOME/lib:/sw/external/python/anaconda3/lib:$LD_LIBRARY_PATH
export PATH=$UCX_HOME/bin:$PATH

export MODELS="meta-llama/Llama-3.1-8B"
export RESULT_DIR="./results/8B-prefill-test"
export PD_RATIO_LIST="1,1 2,1 3,1"
export INPUT_OUTPUT_LIST="1000,100 2000,100 100,100 100,500"
export NUM_PROMPTS_LIST="100"
export RPS_LIST="25"

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

srun bash ./benchmark_xpyd.sh