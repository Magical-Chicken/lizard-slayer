import enum
import util

from lizard import client, server


class EventStatus(enum.Enum):
    PENDING = 'pending'
    RUNNING = 'running'
    SUCCESS = 'success'
    FAILURE = 'failure'


class BaseEventType(enum.Enum):
    INVALID_TYPE = 'invalid_type'
    REQ_SHUTDOWN = 'req_shutdown'
    REGISTER_PROG = 'register_prog'


class ClientEventType(BaseEventType):
    REQ_RUN_PROG = 'req_run_prog'


class ServerEventType(BaseEventType):
    pass


class BaseEvent(object):
    """Base event type"""
    event_map = None
    event_map_lock = None

    def __init__(self, event_type, data):
        """
        Init for Event
        :event_type: event type
        :data: event data
        """
        self.event_type = event_type
        self.event_id = util.hex_id()
        self.status = EventStatus.PENDING
        self.result = None
        self.data = data

    def _register_event(self):
        """
        Register event in event map
        :result: result data
        """
        if self.event_map is None or self.event_map_lock is None:
            raise NotImplementedError("Cannot set result on BaseEvent")
        else:
            with self.event_map_lock:
                self.event_map[self.event_id] = self


class ClientEvent(BaseEvent):
    """Client event"""
    event_map = client.CLIENT_EVENT_MAP
    event_map_lock = client.CLIENT_EVENT_MAP_LOCK


class ServerEvent(BaseEvent):
    """Server event"""
    event_map = server.SERVER_EVENT_MAP
    event_map_lock = server.SERVER_EVENT_MAP_LOCK


def get_event_type_by_name(event_type_name, event_type_class):
    """
    get the event type object for the specified event name
    :event_type_name: event type name string
    :returns: instance of ClientEventType or ServerEventType
    """
    if not isinstance(event_type_class, BaseEventType):
        raise ValueError("invalid event type class")
    result = event_type_class.INVALID_TYPE
    if event_type_name in (e.value for e in event_type_class):
        result = event_type_class(event_type_name)
    return result
