#!/bin/bash
#SBATCH --job-name="test nvlink"
#SBATCH --output="./logs/test-nvlink.out"
#SBATCH --error="./logs/test-nvlink.err"
#SBATCH --partition=gpuA100x4,gpuA100x4-interactive,gpuA100x8,gpuA100x8-interactive
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest
#SBATCH --account=bcjw-delta-gpu
#SBATCH -t 00:05:00

# module reset
module load cuda/12.4

# setup
source $nvme_dir/uv/vllm/bin/activate

export UCX_HOME=$HOME/ucx
export UCX_TLS=tcp,cuda_ipc,cuda_copy,sm,self
export UCX_NET_DEVICES=lo

export LD_LIBRARY_PATH=$UCX_HOME/lib:/sw/external/python/anaconda3/lib:$LD_LIBRARY_PATH
export PATH=$UCX_HOME/bin:$PATH

CUDA_VISIBLE_DEVICES=0,1 \
ucx_perftest -m cuda -t tag_bw -w 200 -n 2000 -s 67108864 &

CUDA_VISIBLE_DEVICES=1,0 \
ucx_perftest -m cuda -t tag_bw -w 200 -n 2000 -s 67108864 localhost