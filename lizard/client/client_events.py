import enum
import os

from lizard import client, events, user_prog


class ClientEventType(enum.Enum):
    INVALID_TYPE = 'invalid_type'
    REQ_SHUTDOWN = 'req_shutdown'
    REGISTER_PROG = 'register_prog'
    REQ_RUN_PROG = 'req_run_prog'


def handle_event_register_prog(event):
    """
    handle 'register_prog' event
    data must include 'name', 'checksum' and 'code'
    :event: event to handle
    :returns: event result data if event sucessfully handled
    :raises: ValueError: if program data does not match checksum
    """
    code = event.data['code']
    name = event.data['name']
    checksum = event.data['checksum']
    with client.client_access() as c:
        user_progs_dir = c.user_progs_dir
    prog_dir = os.path.join(user_progs_dir, checksum)
    code_file = os.path.join(prog_dir, user_prog.KERNEL_FILENAME)
    os.mkdir(prog_dir)
    with open(code_file, 'r') as fp:
        fp.write(code)
    program = user_prog.UserProg(name, checksum, code_file)
    program.verify_checksum()
    # FIXME FIXME FIXME
    # set up program build dir and compile it
    with client.client_access() as c:
        c.user_programs[checksum] = program
    return {}


CLIENT_EVENT_HANDLER_MAP = {
    ClientEventType.REGISTER_PROG: handle_event_register_prog,
}


class ClientEvent(events.BaseEvent):
    """Client event"""
    event_map = client.CLIENT_EVENT_MAP
    event_map_lock = client.CLIENT_EVENT_MAP_LOCK
    event_handler_map = CLIENT_EVENT_HANDLER_MAP
