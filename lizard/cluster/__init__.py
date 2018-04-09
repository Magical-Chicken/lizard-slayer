from subprocess import Popen

from lizard import LOG

NAME = 'lizard-slayer'
MODULE_NAME = 'lizard'
USER = 'wlsaidhi'
END = '.cs.utexas.edu'


class Cluster(object):
    """Main cluster object"""

    def __init__(self, args):
        """
        Cluster init
        :args: parsed cmdline args
        """
        self.host_names = []
        self.nodes = {}

        self.args = args

        # read in host names
        with open(args.file) as f:
            for name in f:
                self.host_names.append(name[:-1])

    def start(self):
        """start cluster"""

        cmd = "cd {}; python3 -m {} client -p {} -a {}".format(
                NAME, MODULE_NAME, self.args.port, self.args.addr)

        # start ut cluster using ssh
        for name in self.host_names[-self.args.count:]:
            args = ['ssh', '-n', USER + '@' + name + END, cmd]
            LOG.debug('Popen arguments %s', args)
            self.nodes[name] = Popen(args)

        # wait for cluster to die
        # for _, p in self.nodes.items():
            # p.wait()

    def kill(self):
        """kills cluster"""
        # FIXME
