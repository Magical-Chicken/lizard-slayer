from lizard import util


class LizardClient(object):
    """Main client object"""

    def __init__(self, args, tmpdir, hardware):
        """
        Client init
        :args: parsed cmdline args
        :tmpdir: temporary directory
        :hardware: hardware info dict
        """
        self.uuid = None
        self.args = args
        self.tmpdir = tmpdir
        self.hardware = hardware
        self.server_url = args.addr + ':' + str(args.port)

    def get(self, endpoint, params=None, expect_json=True, add_uuid=True):
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
        return util.make_api_req(
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
        return util.make_api_req(
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
        util.make_api_req(
            self.server_url, client_url, method='DELETE', expect_json=False)
