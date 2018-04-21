import enum
import os

from lizard import LOG
from lizard import events, server, user_prog


class ServerEventType(enum.Enum):
    INVALID_TYPE = 'invalid_type'
    REQ_SHUTDOWN = 'req_shutdown'
    REGISTER_PROG = 'register_prog'


def handle_event_register_prog(event):
    """
    handle 'register_prog' event
    data must include 'name', 'checksum' and 'code'
    :event: event to handle
    :returns: event result data if event sucessfully handled
    :raises: Exception: if error occurs handling event
    """

    code = event.data['code']
    name = event.data['name']
    checksum = event.data['checksum']

    def callback_func(client, event_props):
        if event_props['status'] != events.EventStatus.SUCCESS.value:
            raise ValueError('{}: failed to register program'.format(client))
        client.registered_progs.append(checksum)

    with server.state_access() as s:
        user_progs_dir = s.user_progs_dir
    code_file = os.path.join(user_progs_dir, checksum)
    with open(code_file, 'w') as fp:
        fp.write(code)
    program = user_prog.UserProg(name, checksum, code_file)
    post_data = event.data.copy()
    post_data['send_remote_event'] = True
    with server.state_access() as s:
        s.post_all('/programs', post_data, callback_func=callback_func)
        s.registered_progs[checksum] = program
    LOG.info('Registered user program: %s', program)
    # FIXME FIXME FIXME
    # should use event wait system to mark prog ready once all clients ready
    return {}


SERVER_EVENT_HANDLER_MAP = {
    ServerEventType.REGISTER_PROG: handle_event_register_prog,
}


class ServerEvent(events.BaseEvent):
    """Server event"""
    event_map = server.SERVER_EVENT_MAP
    event_map_lock = server.SERVER_EVENT_MAP_LOCK
    event_handler_map = SERVER_EVENT_HANDLER_MAP
