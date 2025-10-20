#!/bin/bash
#SBATCH --job-name="xpyd benchmark-8b opt"
#SBATCH --output="./logs/xpyd-benchmark-8b-opt-%j.out"
#SBATCH --error="./logs/xpyd-benchmark-8b-opt-%j.err"
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

export UCX_HOME=$HOME/ucx
export UCX_TLS=tcp,cuda_ipc,cuda_copy,sm,self
export UCX_NET_DEVICES=lo

export LD_LIBRARY_PATH=$UCX_HOME/lib:/sw/external/python/anaconda3/lib:$LD_LIBRARY_PATH
export PATH=$UCX_HOME/bin:$PATH

source $nvme_dir/uv/vllm/bin/activate
export LD_LIBRARY_PATH=/sw/external/python/anaconda3/lib:$LD_LIBRARY_PATH
export UCX_NET_DEVICES=lo

export MODELS="meta-llama/Llama-3.1-8B"
export RESULT_DIR="./results/8B-opt"
export PD_RATIO_LIST="1,1 2,1 3,1 2,2 1,2 1,3"
export INPUT_LENS="1000"
export OUTPUT_LENS="100"
export NUM_PROMPTS="100"
export RPS="25"

srun bash ./benchmark_xpyd_opt.sh