import json
import os
import pkgutil

from lizard import PROGRAM_DATA_DIRNAME
from lizard import util

PROGRAM_SOURCE_FILE_NAMES = {
    'cuda_code': 'kernel.cu',
    'cpp_code': 'wrapper.cpp',
    'python_code': 'python_funcs.py',
}

ADDITIONAL_BUILD_FILES = ('Makefile', 'setup.py')


class UserProg(object):
    """A user program"""

    def __init__(self, name, checksum, data_file, build_dir=None):
        """
        UserProg init
        :name: human readable program name
        :checksum: checksum of data file, and id key
        :data_file: path to json program definition blob
        :build_dir: if provided, directory to build code in
        """
        self.ready = False
        self.name = name
        self.checksum = checksum
        self.build_dir = build_dir
        self.data_file = data_file
        with open(self.data_file, 'r') as fp:
            self.data = json.load(fp)

    def _unpack(self):
        """unpack program files and set up build dir structure"""
        for code_key, filename in PROGRAM_SOURCE_FILE_NAMES.items():
            code = self.data['code'][code_key]
            path = os.path.join(self.build_dir, filename)
            with open(path, 'w') as fp:
                fp.write(code)
        for build_file in ADDITIONAL_BUILD_FILES:
            resource_path = os.path.join(
                PROGRAM_DATA_DIRNAME, 'build_files', build_file)
            data = pkgutil.get_data('lizard', resource_path)
            path = os.path.join(self.build_dir, build_file)
            with open(path, 'wb') as fp:
                fp.write(data)

    def build(self):
        """
        build the shared object and python wrapper module
        note that the build dir must exist and have user prog kernel in it
        """
        if not self.build_dir or not os.path.isdir(self.build_dir):
            raise ValueError("Build dir not set up")
        self._unpack()
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
