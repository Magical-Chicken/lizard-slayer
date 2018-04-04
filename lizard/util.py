import os
import subprocess


def subp(cmd, check=True):
    """
    run a subprocess
    :cmd: cmd to run as list of argv
    :check: if true, check return 0
    :returns: tuple of stdout and stderr
    """
    res = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)
    return res.stdout.decode(), res.stderr.decode()


def construct_sane_url(*args, prefix='http://'):
    """
    construct a url from several components
    :*args: url components
    :prefix: transfer protocol to use in result
    """
    prefixes = ('http://', 'https://')
    res = os.path.join(*(part.lower().strip().lstrip('/') for part in args))
    res = res[next((len(p) for p in prefixes if res.startswith(p)), 0):]
    return prefix + os.path.normpath(res)
