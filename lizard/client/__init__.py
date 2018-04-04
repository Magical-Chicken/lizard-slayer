import json
import requests
from lizard import util


def make_api_req(server_url, endpoint, params={}, raise_for_status=True):
    """
    make an api request
    :server_url: base url for api server
    :endpoint: api endpoint
    :params: get parameters
    :raise_for_status: if true, raise error if status not 200
    :returns: parsed json data from api endpoint
    :raises: OSError: if raise_for_status=True and bad response code
    """
    url = util.construct_sane_url(server_url, endpoint)
    res = requests.get(url, params=params)
    if raise_for_status:
        res.raise_for_status()
    return res.json()


class LizardClient(object):
    """Main client object"""

    def __init__(self, args, hardware):
        """
        Client init
        :args: parsed cmdline args
        :hardware: hardware info dict
        """
        self.args = args
        self.hardware = hardware
        self.server_url = args.addr + ':' + args.port

    def register(self):
        """register client with server"""
        params = {'hardware': json.dumps(self.hardware)}
        make_api_req(self.server_url, '/register', params=params)
