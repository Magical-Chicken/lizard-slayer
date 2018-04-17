import enum

from lizard import client, events


class ClientEventType(enum.Enum):
    INVALID_TYPE = 'invalid_type'
    REQ_SHUTDOWN = 'req_shutdown'
    REGISTER_PROG = 'register_prog'
    REQ_RUN_PROG = 'req_run_prog'


def handle_event_register_prog(event):
    """
    handle a register program event
    :event: event to handle
    :returns: event result data if event sucessfully handled
    :raises: Exception: if error occurs handling event
    """
    raise NotImplementedError


CLIENT_EVENT_HANDLER_MAP = {
    ClientEventType.REGISTER_PROG: handle_event_register_prog,
}


class ClientEvent(events.BaseEvent):
    """Client event"""
    event_map = client.CLIENT_EVENT_MAP
    event_map_lock = client.CLIENT_EVENT_MAP_LOCK
    event_handler_map = CLIENT_EVENT_HANDLER_MAP
