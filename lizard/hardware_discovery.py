import ctypes
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


class GPUProps(ctypes.Structure):
    """GPU properties struct"""
    _fields_ = [
        ('gpu_index', ctypes.c_int),
        ('comp_level_major', ctypes.c_int),
        ('comp_level_minor', ctypes.c_int),
        ('sm_count', ctypes.c_int),
        ('max_sm_threads', ctypes.c_int),
        ('max_sm_blocks', ctypes.c_int),
        ('max_block_size', ctypes.c_int),
        ('max_total_threads', ctypes.c_int),
        ('max_total_blocks', ctypes.c_int),
        ('name', ctypes.c_char * 256),
    ]


def setup_cuda_detect(args, tmpdir):
    """
    set up CUDA detect program
    :args: parsed cmdline args
    :tmpdir: temporary directory
    :returns: wrapped program
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
        'Hardware Discovery', checksum, data_file, {}, build_dir=prog_dir)
    program.build(
        cuda_bin=args.bin, include_path=args.include, unpack=False,
        set_compute_level=False)
    so_path = os.path.join(prog_dir, 'user_program_cuda.so')
    wrapper = ctypes.cdll.LoadLibrary(so_path)
    wrapper.get_num_gpus.restype = ctypes.c_int
    wrapper.get_gpu_data.argtypes = [ctypes.c_int, ctypes.POINTER(GPUProps)]
    return wrapper


def get_reasonable_block_size(props, size_mult=32):
    """
    get reasonable cuda block size
    :props: gpu properties dict
    :size_mult: block size multiple
    :returns: reasonable block size
    """
    max_reasonable_size = props['max_block_size']
    min_reasonable_size = props['max_sm_threads'] / props['max_sm_blocks']
    avg_reasonable_size = (max_reasonable_size + min_reasonable_size) / 2
    reasonable_block_size = int(avg_reasonable_size/size_mult) * size_mult
    LOG.debug('Using CUDA block size: %s', reasonable_block_size)
    return reasonable_block_size


def check_gpus(args, tmpdir):
    """
    check for CUDA capable GPUs
    :args: parsed cmdline args
    :tmpdir: temporary directory
    :returns: dict with GPU info
    """
    if args.no_gpu:
        LOG.warning("Not scanning available gpus, running programs will fail")
        return {'num_gpus': 0, 'gpu_info': []}
    LOG.info('Checking CUDA build system')
    program = setup_cuda_detect(args, tmpdir)
    res = {
        'num_gpus': program.get_num_gpus(),
        'gpu_info': [],
    }
    for gpu_index in range(res['num_gpus']):
        props = GPUProps()
        program.get_gpu_data(gpu_index, ctypes.byref(props))
        gpu_info = {
            'gpu_index': props.gpu_index,
            'comp_level_major': props.comp_level_major,
            'comp_level_minor': props.comp_level_minor,
            'sm_count': props.sm_count,
            'max_sm_threads': props.max_sm_threads,
            'max_sm_blocks': props.max_sm_blocks,
            'max_block_size': props.max_block_size,
            'max_total_threads': props.max_total_threads,
            'max_total_blocks': props.max_total_blocks,
            'name': props.name.decode(),
        }
        gpu_info['reasonable_block_size'] = get_reasonable_block_size(gpu_info)
        res['gpu_info'].append(gpu_info)
    return res


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
