#!/usr/bin/env python3
import json
import glob
import re
import sys
import time
import csv
from typing import List, Dict, Any
import statistics
import math


def parse_config(filename: str):
    """Extract configuration from filename like benchmark-8B-1P3D-2000-200.log"""
    match = re.search(r'benchmark-([\w\.]+)-(\w+)-(\w+)-(\w+)(?:-(\d{8}\.\d{6}))?\.log', filename)
    if match:
        groups = match.groups()
        model, config, input_len, output_len = groups[:4]
        timestamp = groups[4] if groups[4] else "00000000.000000"
        return model, config, input_len, output_len, timestamp
    return filename, '', '', '', "00000000.000000"


def write_to_json(folder_path: str, results_dict: Dict[str, Any]) -> None:
    """Write the aggregated results dict to a JSON file. (Helper; not invoked by main.)"""
    json_output_path = f"./{folder_path}/results.json"
    try:
        with open(json_output_path, 'w') as jf:
            json.dump(results_dict, jf, indent=2)
        print(f"Wrote JSON results to {json_output_path}")
    except Exception as e:
        print(f"Failed to write JSON results to {json_output_path}: {e}")


def write_to_csv(folder_path: str, experiments: List[Dict[str, Any]]) -> None:
    csv_path = f"./{folder_path}/results.csv"
    fieldnames = [
        'Run ID', 'Config', 'Input Len', 'Output Len',
        'RPS', 'Num Prompts',
        'Median ITL (ms)', 'Mean ITL', 'P99 ITL',
        'Median TTFT', 'Mean TTFT', 'P99 TTFT',
        'Input Throughput', 'Output Throughput'
    ]
    try:
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i, exp in enumerate(experiments):
                writer.writerow({
                    'Run ID': i + 1,
                    'Config': exp.get('config'),
                    'Input Len': exp.get('input_len'),
                    'Output Len': exp.get('output_len'),
                    'RPS': exp.get('rps'),
                    'Num Prompts': exp.get('num_prompts'),
                    'Median ITL (ms)': exp.get('median_itl_ms'),
                    'Mean ITL': exp.get('mean_itl_ms'),
                    'P99 ITL': exp.get('p99_itl_ms'),
                    'Median TTFT': exp.get('median_ttft_ms'),
                    'Mean TTFT': exp.get('mean_ttft_ms'),
                    'P99 TTFT': exp.get('p99_ttft_ms'),
                    'Input Throughput': exp.get('input_throughput'),
                    'Output Throughput': exp.get('output_throughput'),
                })
        print(f"Wrote CSV results to {csv_path}")
    except Exception as e:
        print(f"Failed to write CSV results to {csv_path}: {e}")


def aggregate_experiments(experiments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group experiments by (config, num_prompts, rps, input_len, output_len).

    For each group collect the per-run metrics:
      - median_itl_ms (p50 ITL)
      - p99_itl_ms (p99 ITL)
      - mean_itl_ms (mean ITL)
      - median_ttft_ms (p50 TTFT)
      - p99_ttft_ms (p99 TTFT)
      - mean_ttft_ms (mean TTFT)

    Then compute the aggregated metrics as the arithmetic mean across runs for
    each of the above metrics. This ensures we use the already-computed
    p50/p99/mean values from each run and simply average them across runs.
    """

    groups: Dict[tuple, Dict[str, Any]] = {}
    for exp in experiments:
        key = (
            exp.get('config'),
            exp.get('num_prompts'),
            exp.get('rps'),
            exp.get('input_len'),
            exp.get('output_len'),
        )
        g = groups.get(key)
        # pull per-run metrics (may be None)
        p50_itl = exp.get('median_itl_ms')
        mean_itl = exp.get('mean_itl_ms')
        p99_itl = exp.get('p99_itl_ms')

        p50_ttft = exp.get('median_ttft_ms')
        mean_ttft = exp.get('mean_ttft_ms')
        p99_ttft = exp.get('p99_ttft_ms')

        if g is None:
            g = {
                'config': key[0],
                'num_prompts': key[1],
                'rps': key[2],
                'input_len': key[3],
                'output_len': key[4],
                'p50_itl_list': [],
                'mean_itl_list': [],
                'p99_itl_list': [],
                'p50_ttft_list': [],
                'mean_ttft_list': [],
                'p99_ttft_list': [],
            }
            groups[key] = g

        def _append_numeric(lst, val):
            try:
                if val is not None:
                    lst.append(float(val))
            except Exception:
                pass

        _append_numeric(g['p50_itl_list'], p50_itl)
        _append_numeric(g['mean_itl_list'], mean_itl)
        _append_numeric(g['p99_itl_list'], p99_itl)

        _append_numeric(g['p50_ttft_list'], p50_ttft)
        _append_numeric(g['mean_ttft_list'], mean_ttft)
        _append_numeric(g['p99_ttft_list'], p99_ttft)

    aggregated: List[Dict[str, Any]] = []

    for g in groups.values():
        def _mean_or_zero(lst: List[float]) -> float:
            return statistics.mean(lst) if lst else 0.0

        aggregated.append({
            'config': g['config'],
            'num_prompts': g['num_prompts'],
            'rps': g['rps'],
            'input_len': g['input_len'],
            'output_len': g['output_len'],
            # store the lists for debugging if needed
            'p50_itl_list': g['p50_itl_list'],
            'mean_itl_list': g['mean_itl_list'],
            'p99_itl_list': g['p99_itl_list'],
            'p50_ttft_list': g['p50_ttft_list'],
            'mean_ttft_list': g['mean_ttft_list'],
            'p99_ttft_list': g['p99_ttft_list'],
            # aggregated (mean across runs)
            'mean_p50_itl_ms': _mean_or_zero(g['p50_itl_list']),
            'mean_mean_itl_ms': _mean_or_zero(g['mean_itl_list']),
            'mean_p99_itl_ms': _mean_or_zero(g['p99_itl_list']),
            'mean_p50_ttft_ms': _mean_or_zero(g['p50_ttft_list']),
            'mean_mean_ttft_ms': _mean_or_zero(g['mean_ttft_list']),
            'mean_p99_ttft_ms': _mean_or_zero(g['p99_ttft_list']),
            'count': max(
                len(g['p50_itl_list']),
                len(g['p50_ttft_list']),
                len(g['p99_itl_list']),
                len(g['p99_ttft_list'])
            ),
        })

    return aggregated


def write_aggregated_csv(folder_path: str, aggregated: List[Dict[str, Any]]) -> None:
    csv_path = f"./{folder_path}/results_aggregated.csv"
    fieldnames = [
        'Config', 'Num Prompts', 'RPS', 'Input Len', 'Output Len', 'Count',
        'Mean P50 TTFT (ms)', 'Mean Mean TTFT (ms)', 'Mean P99 TTFT (ms)',
        'Mean P50 ITL (ms)', 'Mean Mean ITL (ms)', 'Mean P99 ITL (ms)',
        'P50 TTFT List', 'Mean TTFT List', 'P99 TTFT List',
        'P50 ITL List', 'Mean ITL List', 'P99 ITL List'
    ]
    try:
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in aggregated:
                writer.writerow({
                    'Config': row.get('config'),
                    'Num Prompts': row.get('num_prompts'),
                    'RPS': row.get('rps'),
                    'Input Len': row.get('input_len'),
                    'Output Len': row.get('output_len'),
                    'Count': row.get('count'),
                    'Mean P50 TTFT (ms)': row.get('mean_p50_ttft_ms'),
                    'Mean Mean TTFT (ms)': row.get('mean_mean_ttft_ms'),
                    'Mean P99 TTFT (ms)': row.get('mean_p99_ttft_ms'),
                    'Mean P50 ITL (ms)': row.get('mean_p50_itl_ms'),
                    'Mean Mean ITL (ms)': row.get('mean_mean_itl_ms'),
                    'Mean P99 ITL (ms)': row.get('mean_p99_itl_ms'),
                    'P50 TTFT List': json.dumps(row.get('p50_ttft_list', [])),
                    'Mean TTFT List': json.dumps(row.get('mean_ttft_list', [])),
                    'P99 TTFT List': json.dumps(row.get('p99_ttft_list', [])),
                    'P50 ITL List': json.dumps(row.get('p50_itl_list', [])),
                    'Mean ITL List': json.dumps(row.get('mean_itl_list', [])),
                    'P99 ITL List': json.dumps(row.get('p99_itl_list', [])),
                })
        print(f"Wrote aggregated CSV results to {csv_path}")
    except Exception as e:
        print(f"Failed to write aggregated CSV results to {csv_path}: {e}")


def print_aggregated(aggregated: List[Dict[str, Any]]) -> None:
    """Print aggregated results to the terminal for debugging.

    Shows config, input/output lengths, rps, num_prompts, count, P50s and lists.
    """
    print("\nAggregated results (grouped by config, input_len, output_len, rps, num_prompts):")
    hdr = [
        'Config', 'Num Prompts', 'RPS', 'Input Len', 'Output Len', 'Count',
        'Mean P50 TTFT (ms)', 'Mean Mean TTFT (ms)', 'Mean P99 TTFT (ms)',
        'Mean P50 ITL (ms)', 'Mean Mean ITL (ms)', 'Mean P99 ITL (ms)'
    ]
    print('\t'.join(hdr))
    for row in aggregated:
        print(
            f"{row.get('config')}\t{row.get('num_prompts')}\t{row.get('rps')}\t{row.get('input_len')}\t{row.get('output_len')}\t{row.get('count')}\t{row.get('mean_p50_ttft_ms'):.2f}\t{row.get('mean_mean_ttft_ms'):.2f}\t{row.get('mean_p99_ttft_ms'):.2f}\t{row.get('mean_p50_itl_ms'):.2f}\t{row.get('mean_mean_itl_ms'):.2f}\t{row.get('mean_p99_itl_ms'):.2f}"
        )


def main():
    if len(sys.argv) <= 1:
        print("Error: No sys.arg provided")
        sys.exit(1)

    folder = sys.argv[1]
    log_files = glob.glob(f"./{folder}/benchmark-*.log")

    results: Dict[tuple, Dict[str, Any]] = {}
    model = None

    for i, file in enumerate(log_files):
        model, config, input_len, output_len, timestamp = parse_config(file)
        config_tuple = (config, input_len, output_len, timestamp)

        with open(file, 'r') as f:
            data = json.load(f)

        rps = data.get('request_rate', 'N/A')
        num_prompts = data.get('num_prompts', 'N/A')

        # Compute input throughput from total_input_tokens / duration when possible
        input_throughput_calc = None
        total_input_tokens = data.get('total_input_tokens')
        duration = data.get('duration')
        if total_input_tokens is not None and duration is not None:
            try:
                input_throughput_calc = float(total_input_tokens) / float(duration)
            except Exception:
                input_throughput_calc = None

        # Fallbacks if calc not available
        input_throughput_fallback = (
            data.get('request_throughput')
            or data.get('request_goodput')
            or data.get('total_token_throughput')
            or data.get('request_rate')
        )

        # Use sentinel 0 if neither calc nor fallback is available
        input_throughput_value = (
            input_throughput_calc
            if input_throughput_calc is not None
            else (float(input_throughput_fallback) if input_throughput_fallback is not None else 0.0)
        )

        # Store extracted metrics
        results[config_tuple] = {
            'median_itl_ms': data.get('median_itl_ms'),
            'mean_itl_ms': data.get('mean_itl_ms'),
            'p99_itl_ms': data.get('p99_itl_ms'),
            'median_ttft_ms': data.get('median_ttft_ms'),
            'mean_ttft_ms': data.get('mean_ttft_ms'),
            'p99_ttft_ms': data.get('p99_ttft_ms'),
            'rps': rps,
            'timestamp': timestamp,
            'num_prompts': num_prompts,
            'output_throughput': data.get('output_throughput'),
            'input_throughput': input_throughput_value,
        }

    # Print results
    print("Model: ", model)
    print("Timestamp\tConfig\tInput Len\tOutput Len\tRPS\tNum Prompts\tMedian ITL (ms)\tMedian TTFT (ms)")
    print("-" * 90)

    # only storing results after 15/10
    filtered_keys = [k for k in results.keys() if k[3] > "20251019.150000"]
    sorted_keys = sorted(filtered_keys, key=lambda x: (x[1], x[2], results[x]['rps'], results[x]['num_prompts'], x[3], x[0]))

    # Build structured JSON-able results
    dict_results = {
        'model': model if model is not None else None,
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'experiments': [],
    }

    for config_tuple in sorted_keys:
        config = config_tuple[0]
        input_len = config_tuple[1]
        output_len = config_tuple[2]
        timestamp = config_tuple[3]

        itl = results[config_tuple].get('median_itl_ms') or 0.0
        ttft = results[config_tuple].get('median_ttft_ms') or 0.0
        rps = results[config_tuple].get('rps')
        num_prompts = results[config_tuple].get('num_prompts')
        output_throughput = results[config_tuple].get('output_throughput')

        # Ensure input_throughput numeric and use 0 sentinel when unknown
        try:
            input_throughput_numeric = float(results[config_tuple].get('input_throughput', 0.0))
        except Exception:
            input_throughput_numeric = 0.0

        dict_results['experiments'].append({
            'model': model,
            'timestamp': timestamp,
            'config': config,
            'input_len': int(input_len) if isinstance(input_len, str) and input_len.isdigit() else input_len,
            'output_len': int(output_len) if isinstance(output_len, str) and output_len.isdigit() else output_len,
            'rps': rps,
            'num_prompts': num_prompts,
            'median_itl_ms': results[config_tuple].get('median_itl_ms'),
            'mean_itl_ms': results[config_tuple].get('mean_itl_ms'),
            'p99_itl_ms': results[config_tuple].get('p99_itl_ms'),
            'median_ttft_ms': results[config_tuple].get('median_ttft_ms'),
            'mean_ttft_ms': results[config_tuple].get('mean_ttft_ms'),
            'p99_ttft_ms': results[config_tuple].get('p99_ttft_ms'),
            'input_throughput': input_throughput_numeric,
            'output_throughput': output_throughput,
        })

        # Also print the tabular row
        print(f"{timestamp}\t{config}\t{input_len}\t\t{output_len}\t\t{rps}\t{num_prompts}\t{itl:.2f}\t\t{ttft:.2f}")

    # Note: write_to_json is available but not invoked per request

    # Write CSV output
    # write_to_csv(folder, dict_results['experiments'])
    # Aggregate experiments by (config, input_len, output_len, rps, num_prompts)
    aggregated = aggregate_experiments(dict_results['experiments'])
    print_aggregated(aggregated)
    # Also write aggregated CSV (includes per-run lists for verification)
    try:
        write_aggregated_csv(folder, aggregated)
    except Exception as e:
        print(f"Failed to write aggregated CSV: {e}")


if __name__ == "__main__":
    main()
