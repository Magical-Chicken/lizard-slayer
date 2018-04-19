import contextlib
import threading


REMOTE_EVENTS = None
REMOTE_EVENTS_LOCK = threading.Lock()


def create_remote_events():
    """Create remote events object"""
    with REMOTE_EVENTS_LOCK:
        global REMOTE_EVENTS
        REMOTE_EVENTS = RemoteEvents()


@contextlib.contextmanager
def remote_events_access():
    """contextmanager to provide access to remote events"""
    with REMOTE_EVENTS_LOCK:
        yield REMOTE_EVENTS


class RemoteEvents(object):
    """Manages tracking remote events"""

    def __init__(self):
        """RemoteEvents init"""
        self._remote_event_map = {}

    def register_client(self, client_id):
        """
        Set up tracking for new client
        :client_id: client uuid
        :raises: ValueError: if client already registerd
        """
        if client_id in self._remote_event_map:
            raise ValueError("Client already exists")
        self._remote_event_map[client_id] = {}

    def register_event(self, client_id, event_id, event_props):
        """
        Register a new event
        :client_id: client uuid
        :event_id: event uuid
        :event_props: event properties
        :raises: ValueError: if client does not exist
        """
        if client_id not in self._remote_event_map:
            raise ValueError("Client does not exist")
        self._remote_event_map[client_id][event_id] = event_props

    def update_event(self, client_id, event_id, event_props):
        """
        Update an event
        :client_id: client uuid
        :event_id: event uuid
        :event_props: new event properties
        :raises: ValueError: if client or event does not exist
        """
        if client_id not in self._remote_event_map:
            raise ValueError("Client does not exist")
        if event_id not in self._remote_event_map[client_id]:
            raise ValueError("Event does not exist")
        self._remote_event_map[client_id][event_id] = event_props
