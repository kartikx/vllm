#!/usr/bin/env python3
import json
import glob
import re
import sys

def parse_config(filename):
    """Extract configuration from filename like benchmark-8B-1P3D-2000-200.log"""
    match = re.search(r'benchmark-([\w\.]+)-(\w+)-(\w+)-(\w+)(?:-(\d{8}\.\d{6}))?\.log', filename)
    if match:
        groups = match.groups()
        model, config, input_len, output_len = groups[:4]
        timestamp = groups[4] if groups[4] else "00000000.000000"  # Sentinel value for default timestamp
        return model, config, input_len, output_len, timestamp
    return filename, '', '', '', "00000000.000000"

def main():
    if (len(sys.argv) <= 1):
        print("Error: No sys.arg provided");
        sys.exit(1)

    folder = sys.argv[1]

    log_files = glob.glob(f"./{folder}/benchmark-*.log")

    results = {}

    global model

    for i, file in enumerate(log_files):
        model, config, input_len, output_len, timestamp = parse_config(file)

        config_tuple = (config, input_len, output_len, timestamp)

        with open(file, 'r') as f:
            data = json.load(f)
            rps = data.get('request_rate', 'N/A')
            num_prompts = data.get('num_prompts', 'N/A')

        results[config_tuple] = {
            'median_itl_ms': data['median_itl_ms'],
            'median_ttft_ms': data['median_ttft_ms'],
            'rps': rps,
            'timestamp': timestamp,
            'num_prompts': num_prompts
        }

    # Print results
    print("Model: ", model)
    print("Timestamp\tConfig\tInput Len\tOutput Len\tRPS\tNum Prompts\tMedian ITL (ms)\tMedian TTFT (ms)")
    print("-" * 90)
    for config_tuple in sorted(results.keys(), key=lambda x: (x[3], x[1], x[0])):  # Sort by timestamp, input_len, config
        timestamp, config, input_len, output_len = config_tuple[3], config_tuple[0], config_tuple[1], config_tuple[2]
        itl = results[config_tuple]['median_itl_ms']
        ttft = results[config_tuple]['median_ttft_ms']
        rps = results[config_tuple]['rps']
        num_prompts = results[config_tuple]['num_prompts']
        print(f"{timestamp}\t{config}\t{input_len}\t\t{output_len}\t\t{rps}\t{num_prompts}\t{itl:.2f}\t\t{ttft:.2f}")

if __name__ == "__main__":
    main()
