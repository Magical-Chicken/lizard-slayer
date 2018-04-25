import base64
import hashlib
import json
import os
import requests
import shutil
import socket
import subprocess
import tempfile
import uuid
from contextlib import closing

from lizard import LOG


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


def b64_json_enc(data):
    """
    encode data to b64 encoded json
    :data: data to encode
    :returns: encoded str
    """
    json_str = json.dumps(data)
    return base64.b64encode(json_str.encode()).decode()


def b64_json_dec(encoded):
    """
    decode data from b64 encoded json
    :encoded: encoded str
    :returns: data dict
    """
    json_str = base64.b64decode(encoded).decode()
    return json.loads(json_str)


def checksum(data):
    """
    calculate checksum of data
    :data: bytes object or str
    """
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha1(data).hexdigest()


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


def make_api_req(
        server_url, endpoint, method='GET', data=None, params=None,
        raise_for_status=True, expect_json=True):
    """
    make an api request
    :server_url: base url for api server
    :endpoint: api endpoint
    :method: http method to use, one of GET, POST, PUT, DELETE
    :data: post data dict
    :params: get parameters
    :raise_for_status: if true, raise error if status not 200
    :expect_json: if true, decode response as json
    :returns: parsed json data from api endpoint
    :raises: OSError: if raise_for_status=True and bad response code
    """
    url = construct_sane_url(server_url, endpoint)
    if method == 'GET':
        res = requests.get(url, params=params)
    elif method == 'POST':
        res = requests.post(url, json=data)
    elif method == 'PUT':
        res = requests.put(url, json=data)
    elif method == 'DELETE':
        res = requests.delete(url)
    else:
        raise ValueError('unknown request method')
    if raise_for_status:
        res.raise_for_status()
    return res.json() if expect_json else res.text


class TempDir(object):
    """Temporary directory context manager that can keep tmpdir if needed"""

    def __init__(self, preserve=False, prefix='lizard_slayer_'):
        """
        Init for TempDir
        :preserve: preserve tempdir
        :prefix: tmpdir name prefix
        """
        self.tempdir = None
        self.preserve = preserve
        self.prefix = prefix

    def __enter__(self):
        """
        Create tempdir
        :returns: tempdir path
        """
        self.tempdir = tempfile.mkdtemp(prefix=self.prefix)
        LOG.debug('Using tempdir: %s', self.tempdir)
        return self.tempdir

    def __exit__(self, etype, value, trace):
        """
        Destroy tempdir if no errors occurred or preserve set
        :etype: exception type
        :value: exception value
        :trace: exception traceback
        """
        if etype or self.preserve:
            LOG.info('Preserving tempdir: %s', self.tempdir)
        else:
            shutil.rmtree(self.tempdir)
