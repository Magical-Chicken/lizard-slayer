import os
import sys

try:
    import python_funcs
except ImportError:
    current_dir = os.path.dirname(__file__)
    sys.path.append(os.path.abspath(current_dir))
    import python_funcs

C_EXTENSION = True

def init_global_params(run_settings):
    """
    initialize global parameters
    :run_settings: run settings dict, must contain
        - dims
        - max_iterations
        - num_centroids
        - threshold
    :returns: GlobalParams list
    """
    global_params = []
    global_params.append(run_settings['num_points'])
    global_params.append(run_settings['dims'])
    global_params.append(run_settings['num_centroids'])
    global_params.append(run_settings['max_iterations'])
    global_params.append(run_settings['threshold'])
    return global_params


def init_dataset(global_params, run_settings):
    """
    initialize dataset
    :global_params: global parameters object
    :run_settings: run settings dict, must contain
        - input_file
    :returns: data
    """
    with open(run_settings['input_file'], 'r') as fp:
        data = fp.read()
    return data


def print_result(global_params, end_global_state, quiet_print=False):
    """
    print results from end global state
    :global_params: global parameters object
    :end_global_state: end global state string
    :quiet_print: if true, print result, but not in full detail
    """
    if not quiet_print:
        print("Centroids: \n{}".format(end_global_state))
    # print("Iterations: {}".format(global_state.iteration))
