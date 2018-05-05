import os
import sys

try:
    import python_funcs
except ImportError:
    current_dir = os.path.dirname(__file__)
    sys.path.append(os.path.abspath(current_dir))
    import python_funcs


def init_global_params(run_settings):
    """
    initialize global parameters
    :run_settings: run settings dict, must contain
        - dims
        - max_iterations
    :returns: GlobalParams object
    """
    pass


def init_dataset(global_params, run_settings):
    """
    initialize dataset
    :global_params: global parameters object
    :run_settings: run settings dict, must contain
        - input_file
    :returns: Dataset object
    """
    with open(run_settings['input_file'], 'r') as fp:
        dataset = fp.read()
    return dataset


def print_result(global_params, end_global_state):
    """
    print results from end global state
    :global_params: global parameters object
    :end_global_state: end global state string
    """
    # global_state = python_funcs.GlobalState()
    # global_state.decode(end_global_state, global_params)
    print(global_state)
