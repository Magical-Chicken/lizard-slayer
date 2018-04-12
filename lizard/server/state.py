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

    @property
    def properties(self):
        return {
            'uuid': self.uuid,
            'hardware': self.hardware,
            'url': self.url,
        }


class ServerState(object):
    """Object tracking server state"""

    def __init__(self, args):
        """
        ServerState init
        :args: parsed cmdline args
        """
        self.args = args
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
        LOG.info('registered client: %s at %s', client_uuid, url)
        return client_uuid
