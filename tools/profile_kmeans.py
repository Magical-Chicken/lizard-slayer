#!/usr/bin/python3

import os
import sys
from statistics import mean


try:
    from lizard import runtime_helper
except ImportError:
    parent_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    sys.path.append(os.path.abspath(parent_dir))
    from lizard import runtime_helper


INPUTS_DIR = 'config/user_programs/kmeans/datasets'
RUNNER_MODULE = 'config.user_programs.kmeans.runner'
MAX_ITERATIONS = 50
THRESHOLD = 0.0000001
PROGRAM_INPUTS = (
    {
        'input_file': 'random-n2048-d16-c16.txt',
        'num_points': 2048,
        'dims': 16,
        'num_centroids': 16,
    },
    {
        'input_file': 'random-n16384-d24-c16.txt',
        'num_points': 16384,
        'dims': 24,
        'num_centroids': 16,
    },
    {
        'input_file': 'random-n65536-d32-c16.txt',
        'num_points': 65536,
        'dims': 32,
        'num_centroids': 16,
    },
)


def run_prog(
        server_address, prog_checksum, input_settings,
        max_iterations=MAX_ITERATIONS, threshold=THRESHOLD,
        runner_module=RUNNER_MODULE):
    run_settings = input_settings.copy()
    run_settings['input_file'] = os.path.join(
        INPUTS_DIR, run_settings['input_file'])
    run_settings['max_iterations'] = max_iterations
    run_settings['threshold'] = threshold
    elapsed_time = runtime_helper.run_using_runner_module(
        runner_module, run_settings, server_address, prog_checksum,
        quiet_print=True)
    return elapsed_time


def average_runs(iteration_counts, *run_args, **run_kwargs):
    times = []
    for i in range(iteration_counts + 1):
        print("Running program, iteration: {}".format(i))
        time = run_prog(*run_args, **run_kwargs)
        if i == 0:
            continue
        times.append(time)
    return mean(times)


def run_data_to_csv(path, datapoints):
    row_order = ('time', 'points', 'centroids', 'dims')
    rows = [row_order] + [[str(r[n]) for n in row_order] for r in datapoints]
    fmt = '\n'.join(','.join(row) for row in rows)
    with open(path, 'w') as fp:
        fp.write(fmt)


def run_and_dump(server_address, prog_checksum, output_file, iter_counts=5):
    datapoints = []
    for program_input in PROGRAM_INPUTS:
        print("Running program input: {}".format(program_input))
        time = average_runs(
            iter_counts, server_address, prog_checksum, program_input)
        datapoints.append({
            'time': time,
            'points': program_input['num_points'],
            'centroids': program_input['num_centroids'],
            'dims': program_input['dims'],
        })
    run_data_to_csv(output_file, datapoints)
