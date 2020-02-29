import os
import sys

try:
    import python_funcs
except ImportError:
    current_dir = os.path.dirname(__file__)
    sys.path.append(os.path.abspath(current_dir))
    import python_funcs

C_EXTENSION = False

def init_global_params(run_settings):
    """
    initialize global parameters
    :run_settings: run settings dict, must contain
        - dims
        - max_iterations
    :returns: GlobalParams object
    """
    global_params = python_funcs.GlobalParams()
    global_params.dims = run_settings['dims']
    global_params.max_iterations = run_settings['max_iterations']
    return global_params


def init_dataset(global_params, run_settings):
    """
    initialize dataset
    :global_params: global parameters object
    :run_settings: run settings dict, must contain
        - input_file
    :returns: Dataset object
    """
    dataset = python_funcs.Dataset()
    with open(run_settings['input_file'], 'r') as fp:
        lines = fp.readlines()
    dataset.num_points = len(lines)
    dataset.init_aux_structures(global_params)
    points_ref = dataset.get_ref('points')
    for l_idx in range(dataset.num_points):
        line_vals = [float(v.strip()) for v in lines[l_idx].split(',')]
        for d_idx in range(global_params.dims):
            points_ref[l_idx][d_idx] = line_vals[d_idx]
    return dataset


def print_result(global_params, end_global_state, quiet_print=False):
    """
    print results from end global state
    :global_params: global parameters object
    :end_global_state: end global state string
    :quiet_print: if true, print result, but not in full detail
    """
    global_state = python_funcs.GlobalState()
    global_state.decode(end_global_state, global_params)
    print(global_state.values_aux)
