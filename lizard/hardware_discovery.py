import os
import pkgutil

from lizard import LOG, PROGRAM_DATA_DIRNAME
from lizard import user_prog, util


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


def setup_cuda_detect(args, tmpdir):
    """
    set up CUDA detect program
    :args: parsed cmdline args
    :tmpdir: temporary directory
    :returns: build user_prog
    """
    prog_dir = os.path.join(tmpdir, 'hw_detect')
    os.mkdir(prog_dir)

    def load_resource_to_prog_dir(fname, resource_dir='hw_discovery'):
        resource_path = os.path.join(PROGRAM_DATA_DIRNAME, resource_dir, fname)
        data = pkgutil.get_data('lizard', resource_path)
        path = os.path.join(prog_dir, fname)
        with open(path, 'wb') as fp:
            fp.write(data)
        return data

    for fname in ('kernel.cu', 'program.h'):
        load_resource_to_prog_dir(fname)

    conf_fname = 'config.json'
    data_file = os.path.join(prog_dir, conf_fname)
    conf_raw = load_resource_to_prog_dir(conf_fname)
    checksum = util.checksum(conf_raw)
    program = user_prog.UserProg(
        'Hardware Discovery', checksum, data_file, build_dir=prog_dir)
    program.build(cuda_bin=args.bin, include_path=args.include, unpack=False)
    return program


def check_gpus(args, tmpdir):
    """
    check for CUDA capable GPUs
    :args: parsed cmdline args
    :tmpdir: temporary directory
    :returns: dict with GPU info
    """
    empty_result = {'gpus_present': 0}
    if args.no_gpu:
        LOG.warning("Not scanning available gpus, running programs will fail")
        return empty_result
    LOG.info('Checking CUDA build system')
    program = setup_cuda_detect(args, tmpdir)
    # FIXME FIXME FIXME
    return empty_result


def scan_hardware(args, tmpdir):
    """
    scan system hardware
    :args: parsed cmdline args
    :tmpdir: temporary directory
    :returns: dict with hardware info
    """
    hardware = {
        'CPU': check_cpus(),
        'GPU': check_gpus(args, tmpdir),
    }
    LOG.debug('hardware scan found: %s', hardware)
    return hardware
