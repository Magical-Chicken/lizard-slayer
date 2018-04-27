import enum
import os
import threading

from lizard import LOG
from lizard import events, server, user_prog, util
from lizard.server import remote_event


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
    init_path = os.path.join('/runtimes', prog_checksum, runtime_id)
    wakeup_ev = threading.Event()

    def multi_callback_wakeup(event_props):
        wakeup_ev.set()

    def runtime_init_callback(client, event_props):
        if event_props['status'] != events.EventStatus.SUCCESS.value:
            raise ValueError('{}: error on prog runtime init'.format(client))

    with server.state_access() as s:
        program = s.registered_progs[prog_checksum]
    if not program.ready:
        raise ValueError('cannot run program, not ready')
    runtime = program.get_new_server_runtime(runtime_id)
    runtime.prepare_datastructures(global_params_enc)
    runtime.partition_data(dataset_enc)
    # FIXME FIXME FIXME
    # get this information from data partitions
    client_datasets = {}
    client_datasets_enc = {c: d.encode() for c, d in client_datasets.items()}
    runtime_init_remote_event_ids = []
    for client_uuid, dataset_enc in client_datasets_enc.items():
        data = {
            'runtime_id': runtime_id,
            'checksum': prog_checksum,
            'dtaaset_enc': dataset_enc,
            'global_params_enc': global_params_enc,
            'send_remote_event': True,
        }
        with server.state_access() as s:
            c = s.clients[client_uuid]
            res = c.post(init_path, data, callback_func=runtime_init_callback)
            runtime_init_remote_event_ids.append(res['event_id'])
    with remote_event.remote_events_access() as r:
        r.register_multi_callback(
            runtime_init_remote_event_ids, multi_callback_wakeup)
    wakeup_ev.wait(timeout=300)
    wakeup_ev.clear()
    LOG.info('Runtime initialized for user program: %s', program)

    # FIXME FIXME FIXME
    # set up aggregation result
    # send out run iteration events
    # aggregate run iteration responses
    # run global state update function
    # reset
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
    prog_dir = os.path.join(user_progs_dir, checksum)
    data_file = os.path.join(prog_dir, 'data.json')
    os.mkdir(prog_dir)
    with open(data_file, 'w') as fp:
        fp.write(data)
    program = user_prog.UserProg(name, checksum, data_file, build_dir=prog_dir)
    program.build_for_server()
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
