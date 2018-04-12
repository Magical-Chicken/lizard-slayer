import requests
from lizard import util


def make_api_req(
        server_url, endpoint, method='GET', data=None, params=None,
        raise_for_status=True, expect_json=True):
    """
    make an api request
    :server_url: base url for api server
    :endpoint: api endpoint
    :method: http method to use, one of GET, POST, DELETE
    :data: post data dict
    :params: get parameters
    :raise_for_status: if true, raise error if status not 200
    :expect_json: if true, decode response as json
    :returns: parsed json data from api endpoint
    :raises: OSError: if raise_for_status=True and bad response code
    """
    url = util.construct_sane_url(server_url, endpoint)
    if method == 'GET':
        res = requests.get(url, params=params)
    elif method == 'POST':
        res = requests.post(url, json=data)
    elif method == 'DELETE':
        res = requests.delete(url)
    else:
        raise ValueError('unknown request method')
    if raise_for_status:
        res.raise_for_status()
    return res.json() if expect_json else res.text


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

    def get(self, endpoint, params, expect_json=True, add_uuid=True):
        """
        make a GET request to the server, auto add client uuid to params
        :endpoint: server api endpoint
        :params: GET parameters
        :expect_json: if true, decode response as json
        :add_uuid: if true add uuid to params
        :returns: result data
        :raises: OSError: if bad response code
        """
        if add_uuid:
            params['client_uuid'] = self.uuid
        return make_api_req(
            self.server_url, endpoint, method='GET', params=params,
            expect_json=expect_json)

    def post(self, endpoint, data, expect_json=True, add_uuid=True):
        """
        make a POST request to the server, auto add client uuid to data
        :endpoint: server api endpoint
        :data: data to post as json, must be dict
        :expect_json: if true, decode response as json
        :add_uuid: if true add uuid to params
        :returns: result data
        :raises: OSError: if bad response code
        """
        if add_uuid:
            data['client_uuid'] = self.uuid
        return make_api_req(
            self.server_url, endpoint, method='POST', data=data,
            expect_json=expect_json)

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
        res = self.post('/clients', register_data, add_uuid=False)
        self.uuid = res['uuid']

    def shutdown(self):
        """notify the server that the client is shutting down"""
        client_url = '/clients/{}'.format(self.uuid)
        make_api_req(
            self.server_url, client_url, method='DELETE', expect_json=False)
