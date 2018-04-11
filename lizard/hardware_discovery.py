import os

from lizard import util


def check_cpus():
    """
    check CPU information
    :returns: dict with CPU info
    """
    data = {}
    lscpu_out = util.subp(["lscpu"])[0]
    for k, v in [l.split(':') for l in lscpu_out.splitlines() if ':' in l]:
        data[k.strip()] = v.strip()
    return {'max_threads': int(data['CPU(s)']), 'name': data['Model name']}


def check_gpus(args):
    """
    check for CUDA capable GPUs
    :args: parsed cmdline args
    :returns: dict with GPU info
    """
    data = {
        'gpus_present': 0,
        'gpus': [],
    }
    query_vals = ('name', 'serial', 'index', 'compute_mode', 'memory.total')
    nvidia_smi_path = os.path.join(args.bin, 'nvidia-smi')
    nvidia_smi_out = util.subp([
        nvidia_smi_path, "--format=csv,noheader",
        "--query-gpu={}".format(','.join(query_vals))])[0]
    for line in nvidia_smi_out.splitlines():
        gpu_data = {}
        for idx, val in enumerate(line.split(',')):
            gpu_data[query_vals[idx]] = val
        data['gpus'].append(gpu_data)
    data['gpus_present'] = len(data['gpus'])
    return data


def scan_hardware(args):
    """
    scan system hardware
    :args: parsed cmdline args
    :returns: dict with hardware info
    """
    return {
        'CPU': check_cpus(),
        'GPU': check_gpus(args),
    }
