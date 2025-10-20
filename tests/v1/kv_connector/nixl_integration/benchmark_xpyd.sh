#!/bin/bash
set -e

# Models to run
MODELS=${MODELS:-"meta-llama/Llama-3.1-8B"}

# Where results are stored, update if you want fresh results that don't over-write previous ones.
RESULT_DIR=${RESULT_DIR:-"results/8B-A100"}

# Benchmark configuration - (prefill, decode) instance pairs
PD_RATIO_LIST=${PD_RATIO_LIST:-"1,1 2,1 3,1"}

# Input and output lengths for benchmarks
INPUT_OUTPUT_LIST=${INPUT_OUTPUT_LIST:-"100,100"}
NUM_PROMPTS=${NUM_PROMPTS_LIST:-"100"}
RPS_LIST=${RPS_LIST:-"5"}

# Default value for DECODE_MAX_NUM_BATCHED_TOKENS
DECODE_MAX_NUM_BATCHED_TOKENS=${DECODE_MAX_NUM_BATCHED_TOKENS:-2048}

# Find the git repository root directory
GIT_ROOT=$(git rev-parse --show-toplevel)

SMI_BIN=$(which nvidia-smi || which rocm-smi)

# Trap the SIGINT signal (triggered by Ctrl+C) and ensure cleanup_instances
# is invoked before killing background jobs. Also handle SIGTERM and EXIT.
trap 'cleanup_instances; jobs -pr | xargs -r kill' SIGINT SIGTERM EXIT

# Waits for vLLM to start.
wait_for_server() {
    local port=$1
    timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
        sleep 1
    done" && return 0 || return 1
}

# Function to clean up previous instances
cleanup_instances() {
    echo "Cleaning up any running vLLM instances and proxy server..."
    pkill -f "vllm serve" || true
    pkill -f "toy_proxy_server.py" || true
    sleep 2
}

# Handle to get model-specific arguments for deepseek
get_model_args() {
    local model_name=$1
    local extra_args=""

    if [[ "$model_name" == "deepseek-ai/deepseek-vl2-tiny" ]]; then
        extra_args="--hf_overrides '{\"architectures\": [\"DeepseekVLV2ForCausalLM\"]}' --trust-remote-code"
    fi

    echo "$extra_args"
}

get_num_gpus() {
    if [[ "$SMI_BIN" == *"nvidia"* ]]; then
        echo "$($SMI_BIN --query-gpu=name --format=csv,noheader | wc -l)"
    else
        echo "$($SMI_BIN -l | grep GPU | wc -l)"
    fi
}

run_bench() {
    local model_name=$1
    local prefill_instances=$2
    local decode_instances=$3

    # Use just the size suffix (e.g., 8B) for filenames
    local short_model_name="${model_name##*/}"
    local safe_model_name="${short_model_name##*-}"

    # Use global INPUT_LENS and OUTPUT_LENS variables
    IFS=',' read -ra NUM_PROMPTS_ARRAY <<< "$NUM_PROMPTS_LIST"
    IFS=',' read -ra RPS_ARRAY <<< "$RPS_LIST"

    echo "NUM_PROMPTS_ARRAY: $NUM_PROMPTS_ARRAY"
    echo "RPS_ARRAY: $RPS_ARRAY"

    for input_output_len in $INPUT_OUTPUT_LIST; do
        IFS="," read -r input_len output_len <<< $input_output_len 
        for num_prompts in "${NUM_PROMPTS_ARRAY[@]}"; do
            for rps in "${RPS_ARRAY[@]}"; do
                echo "$input_len $output_len $num_prompts $rps"
                local result_dir=$RESULT_DIR
                # Add timestamp to the result file name
                local timestamp=$(date +%Y%m%d.%H%M%S)
                local result_file="benchmark-${safe_model_name}-${prefill_instances}P${decode_instances}D-${input_len}-${output_len}-${timestamp}.log"
                local full_path="${result_dir}/${result_file}"

                # Ensure the result directory exists
                if [ ! -d "$result_dir" ]; then
                    mkdir -p "$result_dir"
                fi
                
                # Measure the time taken for benchmarking
                local start_time=$(date +%s)

                vllm bench serve --port 8192 --seed $(date +%s) \
                    --model $model_name \
                    --dataset-name random --random-input-len $input_len --random-output-len $output_len \
                    --ignore-eos \
                    --num-prompts $num_prompts --request-rate $rps --save-result --result-dir $result_dir --result-filename $result_file --save-detailed

                local end_time=$(date +%s)
                local elapsed_time=$((end_time - start_time))
                echo "Benchmarking completed in $elapsed_time seconds."
                done
            done
        done
    }

# Function to run tests for a specific model
benchmark_model () {
    local model_name=$1
    local prefill_instances=$2
    local decode_instances=$3

    echo "================================"
    echo "Benchmarking model: $model_name in PD Config [$prefill_instances $decode_instances]"
    echo "================================"

    # Get model-specific arguments
    local model_args=$(get_model_args "$model_name")

    # Arrays to store all hosts and ports
    PREFILL_HOSTS=()
    PREFILL_PORTS=()
    DECODE_HOSTS=()
    DECODE_PORTS=()

    local start_time=$(date +%s)

    # Start prefill instances
    for i in $(seq 0 $((NUM_PREFILL_INSTANCES-1))); do
        # Calculate GPU ID - we'll distribute across available GPUs
        GPU_ID=$((i % $(get_num_gpus)))

        # Calculate port number (base port + instance number)
        PORT=$((8100 + i))
        # Calculate side channel port. Avoid clash with with TP workers. 
        SIDE_CHANNEL_PORT=$((5559 + i))

        echo "Starting prefill instance $i on GPU $GPU_ID, port $PORT"

        # Build the command with or without model-specific args
        BASE_CMD="CUDA_VISIBLE_DEVICES=$GPU_ID VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_PORT vllm serve $model_name \
            --port $PORT \
            --gpu-memory-utilization 0.9 \
            --enforce-eager \
            --kv-transfer-config '{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\"}'"

        if [ -n "$model_args" ]; then
            FULL_CMD="$BASE_CMD $model_args"
        else
            FULL_CMD="$BASE_CMD"
        fi

        eval "$FULL_CMD &"

        # Store host and port for proxy configuration
        PREFILL_HOSTS+=("localhost")
        PREFILL_PORTS+=($PORT)
    done

    # TODO - server logs should be stored somewhere.

    # TODO - runs can collide (same config, diff rps since rps is not in the file name)
    # easy solution - if file exists, sleep and try again with the new timestamp.

    # Start decode instances
    for i in $(seq 0 $((NUM_DECODE_INSTANCES-1))); do
        # Calculate GPU ID - we'll distribute across available GPUs, starting from after prefill GPUs
        GPU_ID=$(((i + NUM_PREFILL_INSTANCES) % $(get_num_gpus)))
        # Calculate port number (base port + instance number)
        PORT=$((8200 + i))
        # Calculate side channel port
        SIDE_CHANNEL_PORT=$((5659 + i))

        echo "Starting decode instance $i on GPU $GPU_ID, port $PORT"

        # Build the command with or without model-specific args
        BASE_CMD="CUDA_VISIBLE_DEVICES=$GPU_ID VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_PORT vllm serve $model_name \
            --port $PORT \
            --max-num-batched-tokens $DECODE_MAX_NUM_BATCHED_TOKENS \
            --enforce-eager \
            --gpu-memory-utilization 0.9 \
            --kv-transfer-config '{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\"}'"

        if [ -n "$model_args" ]; then
            FULL_CMD="$BASE_CMD $model_args"
        else
            FULL_CMD="$BASE_CMD"
        fi

        eval "$FULL_CMD &"

        # Store host and port for proxy configuration
        DECODE_HOSTS+=("localhost")
        DECODE_PORTS+=($PORT)
    done

    # Wait for all instances to start
    for PORT in "${PREFILL_PORTS[@]}"; do
        echo "Waiting for prefill instance on port $PORT to start..."
        wait_for_server $PORT
    done

    for PORT in "${DECODE_PORTS[@]}"; do
        echo "Waiting for decode instance on port $PORT to start..."
        wait_for_server $PORT
    done

    # Measure the time taken to start all Prefill and Decode servers

    # Build the command for the proxy server with all the hosts and ports
    PROXY_CMD="python ${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py --port 8192"

    # Add all prefill hosts and ports
    PROXY_CMD+=" --prefiller-hosts ${PREFILL_HOSTS[@]}"
    PROXY_CMD+=" --prefiller-ports ${PREFILL_PORTS[@]}"

    # Add all decode hosts and ports
    PROXY_CMD+=" --decoder-hosts ${DECODE_HOSTS[@]}"
    PROXY_CMD+=" --decoder-ports ${DECODE_PORTS[@]}"

    # Start the proxy server
    echo "Starting proxy server with command: $PROXY_CMD"
    $PROXY_CMD &

    local end_time=$(date +%s)
    local elapsed_time=$((end_time - start_time))
    echo "All Prefill and Decode servers started in $elapsed_time seconds."

    # Wait for the proxy to start
    sleep 5

    warm_up_server "$model_name"

    echo "server is warmed up ... waiting now"

    # sleep 300_000_000

    # Run lm eval for this model
    # echo "Running tests for $model_name"
    # TEST_MODEL=$model_name python -m pytest -s -x ${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/test_accuracy.py
    run_bench "$model_name" "$prefill_instances" "$decode_instances"

    # Clean up before running next model
    cleanup_instances
    sleep 3
}

# Function to warm up the server before benchmarking
warm_up_server() {
    local model_name=$1

    echo "Warming up the server with 10 requests..."
    vllm bench serve --port 8192 --seed $(date +%s) \
        --model $model_name \
        --dataset-name random --random-input-len 500 --random-output-len 100 \
        --num-prompts 10 --request-rate 2
}

# Function to run benchmarks for different (prefill, decode) configurations
run_benchmark_scenarios() {
    for pd_ratio in $PD_RATIO_LIST; do
        IFS="," read -r NUM_PREFILL_INSTANCES NUM_DECODE_INSTANCES <<< $pd_ratio
        echo "================================"
        echo "Running with $NUM_PREFILL_INSTANCES prefill and $NUM_DECODE_INSTANCES decode instances"
        echo "================================"
        for model in "${MODELS[@]}"; do
            benchmark_model  "$model" "$NUM_PREFILL_INSTANCES" "$NUM_DECODE_INSTANCES"
        done
    done
}

echo "MODELS=$MODELS, PREFILLER_TP_SIZE=$PREFILLER_TP_SIZE, DECODER_TP_SIZE=$DECODER_TP_SIZE, RESULT_DIR=$RESULT_DIR, PD_RATIO_LIST=$PD_RATIO_LIST, INPUT_OUTPUT_LIST=$INPUT_OUTPUT_LIST, NUM_PROMPTS_LIST=$NUM_PROMPTS_LIST, RPS=$RPS_LIST, DECODE_MAX_NUM_BATCHED_TOKENS=$DECODE_MAX_NUM_BATCHED_TOKENS"

# Run benchmarks for different (prefill, decode) configurations
run_benchmark_scenarios

echo "All tests completed!"
