from lizard import util

KERNEL_FILENAME = 'kernel.cu'


class UserProg(object):
    """A user program"""

    def __init__(self, name, checksum, code_file):
        """
        UserProg init
        :name: human readable program name
        :checksum: checksum of code file, and id key
        :code_file: path to code file
        """
        self.name = name
        self.checksum = checksum
        self.code_file = code_file

    @property
    def properties(self):
        return {
            'name': self.name,
            'checksum': self.checksum,
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
