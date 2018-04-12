import os
import socket
import subprocess
import uuid
from contextlib import closing


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


def hex_uuid():
    """get a uuid"""
    return uuid.uuid4().hex


def get_free_port():
    """
    find a free port number
    :port_range: range of port numbers to allocate
    :returns: port number
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(('', 0))
        sock_name = sock.getsockname()
    return sock_name[1]
