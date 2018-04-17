import enum

from lizard import client, events


class ClientEventType(enum.Enum):
    INVALID_TYPE = 'invalid_type'
    REQ_SHUTDOWN = 'req_shutdown'
    REGISTER_PROG = 'register_prog'
    REQ_RUN_PROG = 'req_run_prog'


class ClientEvent(events.BaseEvent):
    """Client event"""
    event_map = client.CLIENT_EVENT_MAP
    event_map_lock = client.CLIENT_EVENT_MAP_LOCK
