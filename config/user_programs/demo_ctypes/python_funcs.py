from ctypes import c_float, c_int, POINTER
from lizard import user_prog_resources


def aggregate(global_params, running_aggregate, aggregation_result):
    """
    add an aggregation result into a running total aggregate
    aggregation must be associative
    :global_params: global parameters object
    :running_aggregate: running top level AggregationResult
    :aggregation_result: new aggregation result to add
    :returns: running aggregate
    """
    running_ref = running_aggregate.get_ref('values')
    agg_ref = aggregation_result.get_ref('values')
    for i in range(global_params.dims):
        running_ref[i] += agg_ref[i]
    return running_aggregate


def init_aggregation_result(global_params, aggregation_result=None):
    """
    init or reset aggregation result
    :global_params: global parameters object
    :aggregation_result: if supplied, aggregation result object
    :returns: aggregation_result param or new AggregationResult object
    """
    if aggregation_result is None:
        aggregation_result = AggregationResult()
        aggregation_result.init_aux_structures(global_params)
    values_ref = aggregation_result.get_ref('values')
    for i in range(global_params.dims):
        values_ref[i] = 0
    return aggregation_result


def init_global_state(global_params, global_state=None):
    """
    initialize global state object
    :global_params: global parameters object
    :global_state: if supplied, global state object
    :returns: global state param or new GlobalState object
    """
    if global_state is None:
        global_state = GlobalState()
        global_state.init_aux_structures(global_params)
    values_ref = global_state.get_ref('values')
    for i in range(global_params.dims):
        values_ref[i] = 0
    global_state.iteration = 0
    return global_state


def update_global_state(global_params, aggregation_result, global_state):
    """
    update global state object
    :global_params: global parameters object
    :aggregation_result: aggregation result
    :global_state: global state object to update
    :returns: global_state param
    """
    global_state.iteration += 1
    values_ref = global_state.get_ref('values')
    agg_res_values_ref = aggregation_result.get_ref('values')
    for i in range(global_params.dims):
        values_ref[i] = agg_res_values_ref[i]
    if global_state.iteration >= global_params.max_iterations:
        global_state.done = True
    return global_state


def get_dataset_slice(
        global_params, full_dataset, start_idx, size, dataset=None):
    """
    get a slice of a dataset
    :global_params: global parameters object
    :full_dataset: full dataset object
    :start_idx: offset into dataset
    :size: number of elements to include
    :dataset: if specified, dataset object
    :returns: dataset param or new Dataset
    :raises: IndexError: if request out of bounds
    """
    if size <= 0 or start_idx + size > full_dataset.num_points:
        raise IndexError
    if dataset is None:
        dataset = Dataset()
    dataset.num_points = size
    dataset.init_aux_structures(global_params)
    full_points_ref = full_dataset.get_ref('points')
    dataset_points_ref = dataset.get_ref('points')
    for i in range(size):
        for j in range(global_params.dims):
            dataset_points_ref[i][j] = full_points_ref[start_idx + i][j]
    return dataset


class GlobalState(user_prog_resources.EncodableStructure):
    # NOTE: all GlobalState objects must have a boolean/int 'done' attribute
    _fields_ = [
        ('done', c_int),
        ('iteration', c_int),
        ('values', POINTER(c_float)),
    ]
    aux_field_names = ('values',)

    def init_aux_structures(self, global_params):
        """
        initialize auxilary datastructures
        :global_params: global paramters object
        """
        self.values_aux = [0] * global_params.dims

    def encode_dynamic(self, global_params):
        """
        encode dynamic alloced data that cannot be auto encoded
        :global_params: global parameters object
        :returns: b64 encoded json
        """
        if self.values:
            values = []
            for i in range(global_params.dims):
                values.append(self.values[i])
        else:
            values = self.values_aux
        return user_prog_resources.b64_json_enc({'values': values})

    def decode_dynamic(self, encoded, global_params):
        """
        decode dynamic alloced data that cannot be auto encoded
        :global_params: global parameters object
        :encoded: b64 encoded json
        """
        decoded = user_prog_resources.b64_json_dec(encoded)
        if self.values:
            for i in range(global_params.dims):
                self.values[i] = decoded['values'][i]
        else:
            self.values_aux = decoded['values']


class AggregationResult(user_prog_resources.EncodableStructure):
    _fields_ = [
        ('values', POINTER(c_float)),
    ]
    aux_field_names = ('values',)

    def init_aux_structures(self, global_params):
        """
        initialize auxilary datastructures
        :global_params: global paramters object
        """
        self.values_aux = [0] * global_params.dims

    def encode_dynamic(self, global_params):
        """
        encode dynamic alloced data that cannot be auto encoded
        :global_params: global parameters object
        :returns: b64 encoded json
        """
        if self.values:
            values = []
            for i in range(global_params.dims):
                values.append(self.values[i])
        else:
            values = self.values_aux
        return user_prog_resources.b64_json_enc({'values': values})

    def decode_dynamic(self, encoded, global_params):
        """
        decode dynamic alloced data that cannot be auto encoded
        :global_params: global parameters object
        :encoded: b64 encoded json
        """
        decoded = user_prog_resources.b64_json_dec(encoded)
        if self.values:
            for i in range(global_params.dims):
                self.values[i] = decoded['values'][i]
        else:
            self.values_aux = decoded['values']


class Dataset(user_prog_resources.EncodableStructure):
    _fields_ = [
        ('num_points', c_int),
        ('points', POINTER(POINTER(c_float))),
    ]
    aux_field_names = ('points',)

    def __len__(self):
        """
        dataset length, must be defined for all dataset objects
        """
        return self.num_points

    def init_aux_structures(self, global_params):
        """
        initialize auxilary datastructures
        :global_params: global paramters object
        """
        self.points_aux = [
            [0] * global_params.dims for _ in range(self.num_points)
        ]

    def encode_dynamic(self, global_params):
        """
        encode dynamic alloced data that cannot be auto encoded
        :global_params: global parameters object
        :returns: b64 encoded json
        """
        if self.points:
            points = [[]] * self.num_points
            for i in range(self.num_points):
                for j in range(global_params.dims):
                    points[i].append(self.points[i][j])
        else:
            points = self.points_aux
        return user_prog_resources.b64_json_enc({'points': points})

    def decode_dynamic(self, encoded, global_params):
        """
        decode dynamic alloced data that cannot be auto encoded
        :global_params: global parameters object
        :encoded: b64 encoded json
        """
        decoded = user_prog_resources.b64_json_dec(encoded)
        if self.points:
            for i in range(self.num_points):
                for j in range(global_params.dims):
                    self.points[i][j] = decoded['points'][i][j]
        else:
            self.points_aux = decoded['points']


class GlobalParams(user_prog_resources.EncodableStructure):
    _fields_ = [
        ('dims', c_int),
        ('max_iterations', c_int),
    ]
    # NOTE: GlobalParams objects cannot have dynamically allocated data fields
