import json
import os
import pkgutil

from lizard import LOG, PROGRAM_DATA_DIRNAME
from lizard import util

PROGRAM_SOURCE_FILE_NAMES = {
    'cuda': 'kernel.cu',
    'cpp': 'wrapper.cpp',
    'python': 'python_funcs.py',
    'header': 'program.h',
}

ADDITIONAL_BUILD_FILES = ('Makefile', 'setup.py')


class UserProg(object):
    """A user program"""

    def __init__(
            self, name, checksum, data_file, build_dir=None,
            ignore_data_file=False):
        """
        UserProg init
        :name: human readable program name
        :checksum: checksum of data file, and id key
        :data_file: path to json program definition blob
        :build_dir: if provided, directory to build code in
        :ignore_data_file: if true, ignore json data file
        """
        self.ready = False
        self.name = name
        self.checksum = checksum
        self.build_dir = build_dir
        self.data_file = data_file
        if not ignore_data_file:
            with open(self.data_file, 'r') as fp:
                self.data = json.load(fp)

    def unpack(self):
        """unpack program files and set up build dir structure"""
        LOG.debug('Extracting user program code')
        for code_key, filename in PROGRAM_SOURCE_FILE_NAMES.items():
            code = self.data['code'][code_key]
            path = os.path.join(self.build_dir, filename)
            with open(path, 'w') as fp:
                fp.write(code)

    def copy_build_files(self, build_files=ADDITIONAL_BUILD_FILES):
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

    def build(
            self, cuda_bin=None, include_path=None, unpack=True,
            build_files=ADDITIONAL_BUILD_FILES):
        """
        build the shared object and python wrapper module
        note that the build dir must exist and have user prog kernel in it
        :cuda_bin: path to cuda tools bin
        :include_path: path to cuda include dir
        :unpack: if true, unpack program json
        :build_files: names of files to copy
        """
        if not self.build_dir or not os.path.isdir(self.build_dir):
            raise ValueError("Build dir not set up")
        if unpack:
            self.unpack()
        if build_files:
            self.copy_build_files(build_files=build_files)
        make_cmd = ['make', '-C', self.build_dir]
        if cuda_bin is not None:
            nvcc_path = os.path.join(cuda_bin, 'nvcc')
            make_cmd.append('NVCC={}'.format(nvcc_path))
        if include_path is not None:
            make_cmd.append('CUDA_L64=-L{}'.format(include_path))
        LOG.debug('Building CUDA shared object')
        util.subp(make_cmd)
        # FIXME FIXME FIXME
        # finsih build process

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
