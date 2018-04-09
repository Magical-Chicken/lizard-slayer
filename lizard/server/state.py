from lizard import LOG, util


class ClientState(object):
    """Object tracking client state"""

    def __init__(self, client_uuid, hardware):
        """
        ClientState init
        :client_uuid: client uuid string
        :hardware: hardware info dict
        """
        self.hardware = hardware
        self.uuid = client_uuid

    @property
    def properties(self):
        return {'uuid': self.uuid, 'hardware': self.hardware}


class ServerState(object):
    """Object tracking server state"""

    def __init__(self, args):
        """
        ServerState init
        :args: parsed cmdline args
        """
        self.args = args
        self.clients = {}

    def register_client(self, hardware):
        """
        register a client with the server
        :hardware: hardware info dict
        :returns: client uuid
        """
        client_uuid = util.hex_uuid()
        self.clients[client_uuid] = ClientState(client_uuid, hardware)
        LOG.info('registered client: %s', client_uuid)
        return client_uuid
