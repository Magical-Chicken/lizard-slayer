from py3nvml import py3nvml

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


def check_gpus():
    """
    check for CUDA capable GPUs
    :returns: dict with GPU info
    """
    # NOTE: This fails to get any useful information such as compute level,
    #       available SMs, or even a good representation of the
    #       microarchitecture and model number. This information may require
    #       a new module to be written wrapping C functions to make calls to
    #       CUDA methods directly. For the time being it may be simpler to
    #       parse the name string and decode the model number, and use a lookup
    #       table to retrieve properties for that model
    data = {'gpus': []}
    py3nvml.nvmlInit()
    data['gpus_present'] = py3nvml.nvmlDeviceGetCount()
    for i in range(data['gpus_present']):
        handle = py3nvml.nvmlDeviceGetHandleByIndex(0)
        data['gpus'].append({
            'index': i,
            'name': py3nvml.nvmlDeviceGetName(handle),
            'vram': py3nvml.nvmlDeviceGetMemoryInfo(handle).total,
            'serial': py3nvml.nvmlDeviceGetSerial(handle),
            'brand_id': py3nvml.nvmlDeviceGetBrand(handle),
        })
    py3nvml.nvmlShutdown()
    return data


def scan_hardware(args):
    """
    scan system hardware
    :args: parsed cmdline args
    :returns: dict with hardware info
    """
    return {
        'CPU': check_cpus(),
        'GPU': check_gpus(),
    }
