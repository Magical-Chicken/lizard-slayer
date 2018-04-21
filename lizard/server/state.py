import functools
import os

from lizard import LOG
from lizard import util
from lizard.server import remote_event


class ClientState(object):
    """Object tracking client state"""

    def __init__(self, client_uuid, hardware, url):
        """
        ClientState init
        :client_uuid: client uuid string
        :hardware: hardware info dict
        :url: url of client
        """
        self.registered_progs = []
        self.hardware = hardware
        self.uuid = client_uuid
        self.url = url

    def _register_event_callback(self, api_res, callback_func):
        """
        register a remote event callback function from an api result
        callback function must take 2 args, client and event props
        :api_res: result of respond_create_event() api route on client
        :callback_func: callback function
        """
        event_id = api_res['event_id']
        func = functools.partial(callback_func, self)
        with remote_event.remote_events_access() as r:
            r.register_callback_func(self.uuid, event_id, func)

    def get(self, endpoint, params=None, expect_json=True, callback_func=None):
        """
        make a GET request to this client
        :endpoint: client api endpoint
        :params: GET parameters
        :expect_json: if true, decode response as json
        :callback_func: if set, expect event creation and register callback
        :returns: result data
        :raises: OSError if bad response code
        """
        res = util.make_api_req(
            self.url, endpoint, method='GET', params=params,
            expect_json=expect_json)
        if callback_func is not None:
            self._register_event_callback(res, callback_func)
        return res

    def post(self, endpoint, data, expect_json=True, callback_func=None):
        """
        make a POST request to this client
        :endpoint: client api endpoint
        :data: data to post as json, must be dict
        :expect_json: if true, decode response as json
        :callback_func: if set, expect event creation and register callback
        :returns: result data
        :raises: OSError if bad response code
        """
        res = util.make_api_req(
            self.url, endpoint, method='POST', data=data,
            expect_json=expect_json)
        if callback_func is not None:
            self._register_event_callback(res, callback_func)
        return res

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
        self.registered_progs = {}
        self.user_progs_dir = os.path.join(self.tmpdir, 'user_progs_raw')
        os.mkdir(self.user_progs_dir)

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
        with remote_event.remote_events_access() as r:
            r.register_client(client_uuid)
        LOG.info('Registered client: %s', self.clients[client_uuid])
        return client_uuid

    def _register_multi_callback_from_remote(self, results, callback_func):
        """
        register callback function to run when all results complete
        :results: all client create event results
        :callback_func: callback function to run
        """
        event_ids = [r['event_id'] for r in results.values()]
        with remote_event.remote_events_access() as r:
            r.register_multi_callback(event_ids, callback_func)

    def get_all(
            self, endpoint, params=None, expect_json=True, callback_func=None,
            multi_callback_func=None):
        """
        make a GET request to all clients, does not raise for bad status code
        :endpoint: client api endpoint
        :params: GET parameters
        :expect_json: if true, decode client responses as json
        :callback_func: if set, expect event creation and register callback
        :multi_callback_func: same as callback but after all clients return
        :returns: tuple of successful results dict, failed client uuids
        """
        res_success = {}
        failed_clients = []
        for client_uuid, client in self.clients.items():
            try:
                res = client.get(
                    endpoint, params, expect_json=expect_json,
                    callback_func=callback_func)
                res_success[client_uuid] = res
            except OSError:
                LOG.warning("Client failed event")
                failed_clients.append(client_uuid)
        self._register_multi_callback_from_remote(
            res_success, multi_callback_func)
        return res_success, failed_clients

    def post_all(
            self, endpoint, data, expect_json=True, callback_func=None,
            multi_callback_func=None):
        """
        make a POST request to all clients, does not raise for bad status code
        :endpoint: client api endpoint
        :data: data to post as json, must be dict
        :expect_json: if true, decode client responses as json
        :callback_func: if set, expect event creation and register callback
        :multi_callback_func: same as callback but after all clients return
        :returns: tuple of successful results dict, failed client uuids
        """
        res_success = {}
        failed_clients = []
        for client_uuid, client in self.clients.items():
            try:
                res = client.post(
                    endpoint, data, expect_json=expect_json,
                    callback_func=callback_func)
                res_success[client_uuid] = res
            except OSError:
                LOG.warning("Client failed event")
                failed_clients.append(client_uuid)
        self._register_multi_callback_from_remote(
            res_success, multi_callback_func)
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
