from lizard import LOG
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
    empty_result = {'gpus_present': 0}
    if args.no_gpu:
        LOG.warning("Not scanning available gpus, running programs will fail")
        return empty_result
    # FIXME FIXME FIXME
    return empty_result


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
