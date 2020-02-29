import importlib
import os
import requests
import time

from lizard.events import FINAL_STATES


def poll_for_event_complete(
        server_address, event_id, poll_freq=1, max_time=600):
    """
    poll for an event to complete
    :server_address: full uri for server
    :event_id: event uuid
    :poll_freq: polling frequency (seconds)
    :max_time: max wait time (seconds)
    :returns: event data
    """
    for _ in range(int(max_time / poll_freq)):
        path = os.path.join(server_address, 'events', event_id)
        data = requests.get(path).json()
        if data['status'] in FINAL_STATES:
            break
        time.sleep(poll_freq)
    return data


def run_using_runner_module(
        runner_module_name, run_settings, server_address, checksum,
        quiet_print=False, print_elapsed_time=True):
    """
    run a program using a runner module, see demo program for example
    :runner_module_name: runner module import name
    :run_settings: dict of run settings needed by runner
    :server_address: full uri for server
    :checksum: program checksum/identifier
    :quiet_print: if true, print result, but not in full detail
    :print_elapsed_time: if true, print out run completion time
    :returns: time taken to run program in seconds
    """
    runner_mod = importlib.import_module(runner_module_name)
    global_params = runner_mod.init_global_params(run_settings)
    dataset = runner_mod.init_dataset(global_params, run_settings)
    if runner_mod.C_EXTENSION:
        event_data = {
            'global_params_enc': global_params,
            'dataset_enc': dataset,
        }
    else:
        event_data = {
            'global_params_enc': global_params.encode(None),
            'dataset_enc': dataset.encode(global_params),
        }
    path = os.path.join(server_address, 'runtimes', checksum)
    req_data = requests.post(path, json=event_data).json()
    result_data = poll_for_event_complete(server_address, req_data['event_id'])
    end_global_state = result_data['result']['end_global_state']
    completion_time = result_data['completion_time']
    if print_elapsed_time:
        print("Run completed in {} seconds".format(completion_time))
    runner_mod.print_result(
        global_params, end_global_state, quiet_print=quiet_print)
    return completion_time
