import os
from subprocess import Popen

from lizard import LOG

NAME = 'lizard-slayer'
MODULE_NAME = 'lizard'
END = '.cs.utexas.edu'
LD_LIBRARY_PATH = '/tmp/lizard-slayer/'

# FIXME hardcoded cuda paths for eldar machines
CUDA_BIN = '/opt/cuda-8.0/bin/'
CUDA_INCLUDE = '/opt/cuda-8.0/lib64/'

LD_LIBRARY_PATH = '/tmp/lizard-slayer/'

DEVNULL = open(os.devnull, 'w')


class Cluster(object):
    """Main cluster object"""

    def __init__(self, args, tmp_dir):
        """
        Cluster init
        :args: parsed cmdline args
        """
        self.host_names = []
        self.nodes = {}
        self.args = args
        self.tmp_dir = tmp_dir

        # read in host names
        with open(args.file) as f:
            for name in f:
                self.host_names.append(name[:-1])

    def start(self):
        """start cluster"""

        cmd = "cd {}; LD_LIBRARY_PATH={} python3 -m {} client -p {} -a {} -b {} -i {}".format(
                NAME, LD_LIBRARY_PATH, MODULE_NAME, self.args.port,
                self.args.addr, CUDA_BIN, CUDA_INCLUDE)

        if self.args.verbose:
            cmd = cmd + ' -v'

        LOG.info('log files stored in tmp dir: %s', self.tmp_dir)
        LOG.info('using default cuda paths')
        LOG.info('bin: %s, include: %s', CUDA_BIN, CUDA_INCLUDE)
        # start ut cluster using ssh
        for name in self.host_names[:self.args.count]:
            args = ['ssh', '-n', self.args.user + '@' + name + END, cmd]
            LOG.debug('Popen arguments %s', args)
            LOG.info('starting node: %s', name)
<<<<<<< HEAD
            self.nodes[name] = Popen(args, stdout=DEVNULL, stderr=DEVNULL)
=======
            log = open(os.path.join(self.tmp_dir, name+'.log'), 'w')
            p = Popen(args, stdout=log, stderr=log)
            self.nodes[name] = {'process': p, 'log': log}
>>>>>>> ef9b13b186c1a356f50a36e78ad91a3ccff76392

        LOG.info('cluster running with %i nodes...', self.args.count)

        # wait for cluster to die
<<<<<<< HEAD
        for _, p in self.nodes.items():
            p.wait()
=======
        for _, node in self.nodes.items():
            node['process'].wait()
>>>>>>> ef9b13b186c1a356f50a36e78ad91a3ccff76392

        # cluster has being termianted by server
        LOG.info('cluster terminated by server')

    def kill(self):
        """forcibly terminates cluster"""

        cmd = 'pkill python -c'

<<<<<<< HEAD
        # terminate python programs on the ut cluster using ssh
        for name in self.host_names[-self.args.count:]:
            args = ['ssh', '-n', self.args.user + '@' + name + END, cmd]
            LOG.debug('Popen arguments %s', args)
            LOG.info('killing node: %s', name)
            Popen(args, stdout=DEVNULL, stderr=DEVNULL)

        LOG.info('cluster killed')
=======
        terminating_processes = []
        # terminate python programs on the ut cluster using ssh
        for name in self.nodes:
            args = ['ssh', '-n', self.args.user + '@' + name + END, cmd]
            LOG.debug('Popen arguments %s', args)
            LOG.info('killing node: %s', name)
            log = self.nodes[name]['log']
            terminating_processes.append(Popen(args, stdout=log, stderr=log))

        for p in terminating_processes:
            p.wait()

        # close log files
        for log in [self.nodes[name]['log'] for name in self.nodes]:
            log.close()

        LOG.info('cluster killed')

        if self.args.keep_tmpdir:
            LOG.info('cluster log files persisted in: %s', self.tmp_dir)
>>>>>>> ef9b13b186c1a356f50a36e78ad91a3ccff76392
