import enum
import os
import sys
import threading

from lizard import LOG
from lizard import events, server, user_prog, util


class ServerEventType(enum.Enum):
    INVALID_TYPE = 'invalid_type'
    REQ_SHUTDOWN = 'req_shutdown'
    REGISTER_PROG = 'register_prog'
    RUN_PROGRAM = 'run_program'


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
    iterate_path = os.path.join(init_path, 'iterate')
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
    runtime_init_remote_event_ids = []
    for client_uuid, dataset_enc in runtime.dataset_partitions_encoded.items():
        data = {
            'runtime_id': runtime_id,
            'checksum': prog_checksum,
            'dataset_enc': dataset_enc,
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
    aggregation_lock = threading.Lock()

    def run_iteration_callback(client, event_props):
        if event_props['status'] != events.EventStatus.SUCCESS.value:
            raise ValueError('{}: error running prog iteration'.format(client))
        with aggregation_lock:
            runtime.aggregate(event_props['result']['aggregation_result_enc'])

    while True:
        post_data = {
            'runtime_id': runtime_id,
            'checksum': prog_checksum,
            'global_state_enc': runtime.global_state_encoded,
            'send_remote_event': True,
        }
        with server.state_access() as s:
            s.post_all(
                iterate_path, post_data, callback_func=run_iteration_callback,
                multi_callback_func=multi_callback_wakeup)
        wakeup_ev.wait(timeout=600)
        wakeup_ev.clear()
        runtime.update_global_state()
        runtime.reset_aggregation_result()
        LOG.debug('Completed iteration for user program: %s', program)
        if runtime.done:
            break

    LOG.info('Finished running user program: %s', program)
    return {
        'end_aggregate': runtime.top_level_aggregate_encoded,
        'end_global_state': runtime.global_state_encoded,
    }


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
        all_hardware = s.all_clients_hardware
    prog_dir = os.path.join(user_progs_dir, checksum)
    data_file = os.path.join(prog_dir, 'data.json')
    os.mkdir(prog_dir)
    with open(data_file, 'w') as fp:
        fp.write(data)
    program = user_prog.UserProg(
        name, checksum, data_file, all_hardware, build_dir=prog_dir)
    program.build_for_server()
    post_data = event.data.copy()
    post_data['send_remote_event'] = True
    with server.state_access() as s:
        s.post_all('/programs', post_data, callback_func=callback_func,
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
