import python_funcs


def init_global_params(global_params_conf):
    """
    initialize global parameters
    :global_params_conf: global params config dict, must contain
        - dims
        - max_iterations
    :returns: GlobalParams object
    """
    global_params = python_funcs.GlobalParams
    global_params.dims = global_params_conf['dims']
    global_params.max_iterations = global_params_conf['max_iterations']
    return global_params


def init_dataset(global_params, dataset_conf):
    """
    initialize dataset
    :global_params: global parameters object
    :dataset_conf: dataset config dict, must contain
        - input_file
    :returns: Dataset object
    """
    dataset = python_funcs.Dataset()
    with open(dataset_conf['input_file'], 'r') as fp:
        lines = fp.readlines()
    dataset.num_points = len(lines)
    dataset.init_aux_structures(global_params)
    points_ref = dataset.get_ref('points')
    for l_idx in range(dataset.num_points):
        line_vals = lines[l_idx].split(',')
        for d_idx in range(global_params.dims):
            points_ref[l_idx][d_idx] = line_vals[d_idx]
    return dataset
