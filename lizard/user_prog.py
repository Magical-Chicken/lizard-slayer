import ctypes
import json
import importlib
import os
import pkgutil

from lizard import LOG, PROGRAM_DATA_DIRNAME
from lizard import util

PROGRAM_SHARED_OBJ_NAME = 'user_program_cuda.so'
PROGRAM_SOURCE_FILE_NAMES = {
    'cuda': 'kernel.cu',
    'cpp': 'wrapper.cpp',
    'python': 'python_funcs.py',
    'header': 'program.h',
}


class UserProg(object):
    """A user program"""

    def __init__(self, name, checksum, data_file, build_dir=None):
        """
        UserProg init
        :name: human readable program name
        :checksum: checksum of data file, and id key
        :data_file: path to json program definition blob
        :build_dir: if provided, directory to build code in
        :ignore_data_file: if true, ignore json data file
        """
        self.program_runtimes = {}
        self.server_runtimes = {}
        self.ready = False
        self.name = name
        self.checksum = checksum
        self.build_dir = build_dir
        self.data_file = data_file
        with open(self.data_file, 'r') as fp:
            self.data = json.load(fp)
        self.use_c_extention = self.data['info'].get('py_c_extention', True)

    def get_program_py_mod(self):
        """
        Get python module for the user program
        :returns: python module
        """
        py_name = 'user_program.{}'.format(self.checksum)
        py_file = PROGRAM_SOURCE_FILE_NAMES['python']
        py_path = os.path.join(self.build_dir, py_file)
        py_spec = importlib.util.spec_from_file_location(py_name, py_path)
        py_mod = importlib.util.module_from_spec(py_spec)
        py_spec.loader.exec_module(py_mod)
        return py_mod

    def get_new_program_runtime(self, runtime_id, hardware):
        """
        Get a new program runtime instance
        :runtime_id: program runtime uuid
        :hardware: hardware info dict
        :returns: runtime instance
        """
        if not self.ready:
            raise ValueError("Cannot get program runtime, program not ready")
        if self.use_c_extention:
            raise NotImplementedError
        else:
            path = os.path.join(self.build_dir, PROGRAM_SHARED_OBJ_NAME)
            prog = ctypes.cdll.LoadLibrary(path)
            py_mod = self.get_program_py_mod()
            runtime = UserProgRuntimeCTypes(
                runtime_id, hardware, self.data['info'], prog, py_mod)
        self.program_runtimes[runtime_id] = runtime
        return runtime

    def get_new_server_runtime(self, runtime_id, hardware):
        """
        Get a new server program runtime instance
        :runtime_id: program runtime uuid
        :hardware: all clients hardware info dict
        :returns: runtime instance
        """
        if not self.ready:
            raise ValueError("Cannot get server runtime, program not ready")
        if self.use_c_extention:
            raise NotImplementedError
        else:
            py_mod = self.get_program_py_mod()
            runtime = ServerRuntimeCTypes(
                runtime_id, hardware, self.data['info'], py_mod)
        self.server_runtimes[runtime_id] = runtime
        return runtime

    def unpack(self, item_keys):
        """
        unpack program files and set up build dir structure
        :item_keys: items to unpack
        """
        LOG.debug('Extracting user program code')
        for key in item_keys:
            filename = PROGRAM_SOURCE_FILE_NAMES[key]
            code = self.data['code'][key]
            path = os.path.join(self.build_dir, filename)
            with open(path, 'w') as fp:
                fp.write(code)

    def copy_build_files(self, build_files):
        """
        copy build files from data/build_files into build dir
        :build_files: names of files to copy
        """
        LOG.debug('Copying additional build files')
        for build_file in build_files:
            resource_path = os.path.join(
                PROGRAM_DATA_DIRNAME, 'build_files', build_file)
            data = pkgutil.get_data('lizard', resource_path)
            path = os.path.join(self.build_dir, build_file)
            with open(path, 'wb') as fp:
                fp.write(data)

    def build_for_server(self):
        """
        set up only the user program resources needed for the server
        """
        if not self.build_dir or not os.path.isdir(self.build_dir):
            raise ValueError("Build dir not set up")
        if self.use_c_extention:
            raise NotImplementedError
        self.unpack(['python'])
        self.copy_build_files(['resources.py'])
        self.ready = True

    def build(self, cuda_bin=None, include_path=None, unpack=True):
        """
        set up user program resources and build shared obj
        :cuda_bin: path to cuda tools bin
        :include_path: path to cuda include dir
        :unpack: if true, unpack program json
        """
        if not self.build_dir or not os.path.isdir(self.build_dir):
            raise ValueError("Build dir not set up")
        if unpack:
            files = ['cuda', 'python', 'header']
            if self.use_c_extention:
                files.append('cpp')
            self.unpack(files)
        build_files = ['Makefile']
        if self.use_c_extention:
            build_files.append('setup.py')
        else:
            build_files.append('resources.py')
        self.copy_build_files(build_files)
        make_cmd = ['make', '-C', self.build_dir]
        if cuda_bin is not None:
            nvcc_path = os.path.join(cuda_bin, 'nvcc')
            make_cmd.append('NVCC={}'.format(nvcc_path))
        if include_path is not None:
            make_cmd.append('CUDA_L64=-L{}'.format(include_path))
        LOG.debug('Building CUDA shared object')
        util.subp(make_cmd)
        if self.use_c_extention:
            # FIXME FIXME FIXME
            # finsih build process for c extention
            raise NotImplementedError
        else:
            LOG.debug('No python c extention for user program')
            self.ready = True

    @property
    def properties(self):
        return {
            'name': self.name,
            'checksum': self.checksum,
            'ready': self.ready,
            'info': self.data['info'],
        }

    def verify_checksum(self):
        """
        ensure that the data file matches the program checksum
        :raises: ValueError: if program data does not match checksum
        """
        with open(self.data_file, 'rb') as fp:
            res = util.checksum(fp.read())
        if res != self.checksum:
            raise ValueError("Checksum does not match")

    def __str__(self):
        """String representation for UserProg"""
        return "UserProg: {} checksum: {}".format(self.name, self.checksum)


class ServerRuntimeCTypes(object):
    """Server program runtime for ctypes programs"""

    def __init__(self, runtime_id, hardware, info, py_mod):
        """
        Server runtime init
        :runtime_id: program runtime uuid
        :hardware: all clients hardware info dict
        :info: program conf info
        :py_mod: user program python module
        """
        self.runtime_id = runtime_id
        self.hardware = hardware
        self.py_mod = py_mod
        self.info = info
        self.main_dataset = None
        self.agg_res = None
        self.global_state = None
        self.global_params = None

    def prepare_datastructures(self, global_params_enc):
        """
        prepare user program data structures
        :global_params_enc: encoded global params
        """
        self.global_params = self.py_mod.GlobalParams()
        self.global_params.decode(global_params_enc)
        # FIXME FIXME FIXME
        # set up python accessible datastructures for agg res and global state
        # set up python accessible structure to hold dataset while partitioning

    def partition_data(self, dataset_enc):
        """
        load dataset and partition among clients
        :dataset_enc: encoded data
        """
        raise NotImplementedError


class UserProgRuntimeCTypes(object):
    """User program runtime for ctypes programs"""

    def __init__(self, runtime_id, hardware, info, prog, py_mod):
        """
        User program runtime init
        :runtime_id: program runtime uuid
        :hardware: hardware info struct
        :info: program conf info
        :prog: user program cdll
        :py_mod: user program python module
        """
        self.runtime_id = runtime_id
        self.hardware = hardware
        self.info = info
        self.prog = prog
        self.py_mod = py_mod
        self.dataset = None
        self.agg_res = None
        self.global_state = None
        self.global_params = None
        self.blocks = 0
        self.block_size = 0
        self._configure_functions()

    def load_data(self, dataset_enc):
        """
        load dataset
        :dataset_enc: encoded data
        """
        self.dataset.decode(dataset_enc)
        # FIXME FIXME FIXME
        # calculate number of blocks and block size for processing dataset

    def run_iteration(self, global_state_enc):
        """
        update global state, run iteration, and encode aggregation result
        :global_state_enc: encoded global state
        :returns: encoded aggregation result
        """
        self.global_state.decode(global_state_enc)
        self.prog.run_iteration(
            self.blocks, self.block_size, ctypes.byref(self.global_params),
            ctypes.byref(self.dataset), ctypes.byref(self.global_state),
            ctypes.byref(self.aggregation_result))
        return self.aggregation_result.encode()

    def prepare_datastructures(self, global_params_enc):
        """
        prepare user program data structures
        :global_params_enc: encoded global params
        """
        self.global_params = self.py_mod.GlobalParams()
        self.global_params.decode(global_params_enc)
        self.dataset = self.py_mod.Dataset()
        self.agg_res = self.py_mod.AggregationResult()
        self.global_state = self.py_mod.GlobalState()
        self.prog.setup_dataset(
            ctypes.byref(self.dataset), ctypes.byref(self.global_params))
        self.prog.setup_aggregation_result(
            ctypes.byref(self.agg_res), ctypes.byref(self.global_params))
        self.prog.setup_global_state(
            ctypes.byref(self.global_state), ctypes.byref(self.global_params))

    def free_datastructures(self):
        """free memory allocated for storing program data"""
        self.prog.free_dataset(
            ctypes.byref(self.dataset), ctypes.byref(self.global_params))
        self.prog.free_aggregation_result(
            ctypes.byref(self.agg_res), ctypes.byref(self.global_params))
        self.prog.free_global_state(
            ctypes.byref(self.global_state), ctypes.byref(self.global_params))

    def _configure_functions(self):
        """configure program function arg and res types"""
        self.prog.setup_dataset.argtypes = [
            ctypes.POINTER(self.py_mod.Dataset),
            ctypes.POINTER(self.py_mod.DatasetParams),
        ]
        self.prog.setup_aggregation_result.argtypes = [
            ctypes.POINTER(self.py_mod.AggregationResult),
            ctypes.POINTER(self.py_mod.DatasetParams),
        ]
        self.prog.setup_global_state.argtypes = [
            ctypes.POINTER(self.py_mod.GlobalState),
            ctypes.POINTER(self.py_mod.DatasetParams),
        ]
        self.prog.free_dataset.argtypes = [
            ctypes.POINTER(self.py_mod.Dataset),
            ctypes.POINTER(self.py_mod.DatasetParams),
        ]
        self.prog.free_aggregation_result.argtypes = [
            ctypes.POINTER(self.py_mod.AggregationResult),
            ctypes.POINTER(self.py_mod.DatasetParams),
        ]
        self.prog.free_global_state.argtypes = [
            ctypes.POINTER(self.py_mod.GlobalState),
            ctypes.POINTER(self.py_mod.DatasetParams),
        ]
        self.prog.run_iteration.argtypes = [
            ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(self.py_mod.DatasetParams),
            ctypes.POINTER(self.py_mod.Dataset),
            ctypes.POINTER(self.py_mod.GlobalState),
            ctypes.POINTER(self.py_mod.AggregationResult),
        ]
