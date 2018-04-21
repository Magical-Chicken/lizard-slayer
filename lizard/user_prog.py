import os

from lizard import util

KERNEL_FILENAME = 'kernel.cu'


class UserProg(object):
    """A user program"""

    def __init__(self, name, checksum, code_file, build_dir=None):
        """
        UserProg init
        :name: human readable program name
        :checksum: checksum of code file, and id key
        :code_file: path to code file
        :build_dir: if provided, directory to build code in
        """
        self.ready = False
        self.name = name
        self.checksum = checksum
        self.code_file = code_file
        self.build_dir = build_dir

    def build(self):
        """
        build the shared object and python wrapper module
        note that the build dir must exist and have user prog kernel in it
        """
        kernel_file = os.path.join(self.build_dir, KERNEL_FILENAME)
        if not self.build_dir or not os.path.exists(kernel_file):
            raise ValueError("Build dir is not set up")
        raise NotImplementedError

    @property
    def properties(self):
        return {
            'name': self.name,
            'checksum': self.checksum,
            'ready': self.ready,
        }

    def verify_checksum(self):
        """
        ensure that program was code file matches checksum
        :raises: ValueError: if program data does not match checksum
        """
        with open(self.code_file, 'rb') as fp:
            res = util.checksum(fp.read())
        if res != self.checksum:
            raise ValueError("Code file checksum does not match")

    def __str__(self):
        """String representation for UserProg"""
        return "UserProg: {} checksum: {}".format(self.name, self.checksum)
