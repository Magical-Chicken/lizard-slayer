import requests
from lizard import util


def make_api_req(
        server_url, endpoint, method='GET', data=None, params=None,
        raise_for_status=True):
    """
    make an api request
    :server_url: base url for api server
    :endpoint: api endpoint
    :data: post data dict
    :params: get parameters
    :raise_for_status: if true, raise error if status not 200
    :returns: parsed json data from api endpoint
    :raises: OSError: if raise_for_status=True and bad response code
    """
    url = util.construct_sane_url(server_url, endpoint)
    if method == 'GET':
        res = requests.get(url, params=params)
    elif method == 'POST':
        res = requests.post(url, json=data)
    else:
        raise ValueError('unknown request method')
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
        self.uuid = None
        self.args = args
        self.hardware = hardware
        self.server_url = args.addr + ':' + str(args.port)

    def register(self, client_port):
        """
        register client with server
        :client_port: port number client has bound to
        """
        self.client_port = client_port
        register_data = {
            'hardware': self.hardware,
            'client_port': client_port,
        }
        res = make_api_req(
            self.server_url, '/clients', method='POST', data=register_data)
        self.uuid = res['uuid']
