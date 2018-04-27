import enum
import os
import threading

from lizard import LOG
from lizard import events, server, user_prog, util


class ServerEventType(enum.Enum):
    INVALID_TYPE = 'invalid_type'
    REQ_SHUTDOWN = 'req_shutdown'
    REGISTER_PROG = 'register_prog'
    RUN_PROGRAM = 'RUN_PROGRAM'


def handle_event_run_program(event):
    """
    handle 'run_program' eent
    :event: event to handle
    :returns: program result
    :raises: Exception: if error occurs or invalid request
    """
    runtime_id = util.hex_uuid()
    dataset_enc = event.data['dataset_enc']
    prog_checksum = event.data['checksum']
    global_params_enc = event.data['global_params_enc']
    wakeup_ev = threading.Event()

    def multi_callback_wakeup(event_props):
        wakeup_ev.set()

    with server.state_access() as s:
        program = s.registered_progs[prog_checksum]
    if not program.ready:
        raise ValueError('cannot run program, not ready')
    runtime = program.get_new_server_runtime(runtime_id)
    runtime.prepare_datastructures(global_params_enc)
    runtime.partition_data(dataset_enc)

    # FIXME FIXME FIXME
    # load event data objs and gen events
    # send out init_runtime events
    # wait on init_runtime events
    # set up aggregation result
    raise NotImplementedError


def handle_event_register_prog(event):
    """
    handle 'register_prog' event
    data must include 'name', 'checksum' and 'data'
    :event: event to handle
    :returns: event result data if event sucessfully handled
    :raises: Exception: if error occurs handling event
    """

    data = event.data['data']
    name = event.data['name']
    checksum = event.data['checksum']
    wakeup_ev = threading.Event()

    def multi_callback_func(event_props):
        wakeup_ev.set()

    def callback_func(client, event_props):
        if event_props['status'] != events.EventStatus.SUCCESS.value:
            raise ValueError('{}: failed to register program'.format(client))
        client.registered_progs.append(checksum)

    with server.state_access() as s:
        user_progs_dir = s.user_progs_dir
    data_file = os.path.join(user_progs_dir, checksum)
    with open(data_file, 'w') as fp:
        fp.write(data)
    program = user_prog.UserProg(name, checksum, data_file)
    post_data = event.data.copy()
    post_data['send_remote_event'] = True
    with server.state_access() as s:
        s.post_all(
            '/programs', post_data, callback_func=callback_func,
            multi_callback_func=multi_callback_func)
    # NOTE: timeout for registering program on all nodes set to 10 min
    wakeup_ev.wait(timeout=600)
    LOG.info('Registered user program: %s', program)
    with server.state_access() as s:
        program.ready = True
        s.registered_progs[checksum] = program
    return program.properties


SERVER_EVENT_HANDLER_MAP = {
    ServerEventType.REGISTER_PROG: handle_event_register_prog,
    ServerEventType.RUN_PROGRAM: handle_event_run_program,
}


class ServerEvent(events.BaseEvent):
    """Server event"""
    event_map = server.SERVER_EVENT_MAP
    event_map_lock = server.SERVER_EVENT_MAP_LOCK
    event_handler_map = SERVER_EVENT_HANDLER_MAP
