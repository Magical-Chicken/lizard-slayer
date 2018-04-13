from lizard import LOG, util


class ClientState(object):
    """Object tracking client state"""

    def __init__(self, client_uuid, hardware, url):
        """
        ClientState init
        :client_uuid: client uuid string
        :hardware: hardware info dict
        :url: url of client
        """
        self.hardware = hardware
        self.uuid = client_uuid
        self.url = url

    def get(self, endpoint, params=None, expect_json=True):
        """
        make a GET request to this client
        :endpoint: client api endpoint
        :params: GET parameters
        :expect_json: if true, decode response as json
        :returns: result data
        :raises: OSError if bad response code
        """
        return util.make_api_req(
            self.url, endpoint, method='GET', params=params,
            expect_json=expect_json)

    def post(self, endpoint, data, expect_json=True):
        """
        make a POST request to this client
        :endpoint: client api endpoint
        :data: data to post as json, must be dict
        :expect_json: if true, decode response as json
        :returns: result data
        :raises: OSError if bad response code
        """
        return util.make_api_req(
            self.url, endpoint, method='POST', data=data,
            expect_json=expect_json)

    def delete(self, endpoint):
        """
        make a POST request to this client
        :endpoint: client api endpoint
        :returns: result data
        :raises: OSError if bad response code
        """
        return util.make_api_req(
            self.url, endpoint, method='DELETE', expect_json=False)

    @property
    def properties(self):
        return {
            'uuid': self.uuid,
            'hardware': self.hardware,
            'url': self.url,
        }

    def __str__(self):
        """ClientState string representation"""
        return "Client ID: {} URL: {}".format(self.uuid, self.url)


class ServerState(object):
    """Object tracking server state"""

    def __init__(self, args, tmpdir):
        """
        ServerState init
        :args: parsed cmdline args
        :tmpdir: temporary directory
        """
        self.args = args
        self.tmpdir = tmpdir
        self.clients = {}

    def register_client(self, hardware, client_ip, client_port):
        """
        register a client with the server
        :hardware: hardware info dict
        :client_ip: addr of client
        :client_port: port number for client
        :returns: client uuid
        """
        client_uuid = util.hex_uuid()
        url = 'http://{}:{}'.format(client_ip, client_port)
        self.clients[client_uuid] = ClientState(client_uuid, hardware, url)
        LOG.info('Registered client: %s', self.clients[client_uuid])
        return client_uuid

    def get_all(self, endpoint, params=None, expect_json=True):
        """
        make a GET request to all clients, does not raise for bad status code
        :endpoint: client api endpoint
        :params: GET parameters
        :expect_json: if true, decode client responses as json
        :returns: tuple of successful results dict, failed client uuids
        """
        res_success = {}
        failed_clients = []
        for client_uuid, client in self.clients.items():
            try:
                res = client.get(endpoint, params, expect_json=expect_json)
                res_success[client_uuid] = res
            except OSError:
                failed_clients.append(client_uuid)
        return res_success, failed_clients

    def post_all(self, endpoint, data, expect_json=True):
        """
        make a POST request to all clients, does not raise for bad status code
        :endpoint: client api endpoint
        :data: data to post as json, must be dict
        :expect_json: if true, decode client responses as json
        :returns: tuple of successful results dict, failed client uuids
        """
        res_success = {}
        failed_clients = []
        for client_uuid, client in self.clients.items():
            try:
                res = client.post(endpoint, data, expect_json=expect_json)
                res_success[client_uuid] = res
            except OSError:
                failed_clients.append(client_uuid)
        return res_success, failed_clients

    def delete_all(self, endpoint):
        """
        make a DELETE request to all clients, does not raise for bad status
        :endpoint: client api endpoint
        :returns: tuple of successful uuids, failed client uuids
        """
        success_clients = []
        failed_clients = []
        for client_uuid, client in self.clients.items():
            try:
                client.delete(endpoint)
                success_clients.append(client_uuid)
            except OSError:
                failed_clients.append(client_uuid)
        return success_clients, failed_clients
