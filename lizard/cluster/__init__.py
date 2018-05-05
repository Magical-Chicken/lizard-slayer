import os
from subprocess import Popen

from lizard import LOG

NAME = 'lizard-slayer'
MODULE_NAME = 'lizard'
END = '.cs.utexas.edu'
LD_LIBRARY_PATH = '/tmp/lizard-slayer/'

DEVNULL = open(os.devnull, 'w')


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

        cmd = "cd {}; LD_LIBRARY_PATH={} python3 -m {} client -p {} -a {}".format(
                LD_LIBRARY_PATH, NAME, MODULE_NAME, self.args.port, self.args.addr)

        # start ut cluster using ssh
        for name in self.host_names[:self.args.count]:
            args = ['ssh', '-n', self.args.user + '@' + name + END, cmd]
            LOG.debug('Popen arguments %s', args)
            LOG.info('starting node: %s', name)
            self.nodes[name] = Popen(args, stdout=DEVNULL, stderr=DEVNULL)

        LOG.info('cluster running with %i nodes...', self.args.count)

        # wait for cluster to die
        for _, p in self.nodes.items():
            p.wait()

        # cluster has being termianted by server
        LOG.info('cluster terminated by server')

    def kill(self):
        """forcibly terminates cluster"""

        cmd = 'pkill python -c'

        # terminate python programs on the ut cluster using ssh
        for name in self.host_names[-self.args.count:]:
            args = ['ssh', '-n', self.args.user + '@' + name + END, cmd]
            LOG.debug('Popen arguments %s', args)
            LOG.info('killing node: %s', name)
            Popen(args, stdout=DEVNULL, stderr=DEVNULL)

        LOG.info('cluster killed')
