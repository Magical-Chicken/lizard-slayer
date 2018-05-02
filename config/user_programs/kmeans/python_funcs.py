import math

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
    running_centroids_ref = running_aggregate.get_ref('centroid_updates')
    agg_res_centroids_ref = running_aggregate.get_ref('centroid_updates')
    for i in range(global_params.num_centroids):
        for j in range(global_params.dims):
            running_centroids_ref[i][j] += agg_res_centroids_ref[i][j]
    running_count_ref = running_aggregate.get_ref('update_counts')
    agg_res_count_ref = running_aggregate.get_ref('update_counts')
    for i in range(global_params.num_centroids):
        running_count_ref[i] += agg_res_count_ref[i]
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
    centroid_updates_ref = aggregation_result.get('centroid_updates')
    for i in range(global_params.num_centroids):
        for j in range(global_params.dims):
            centroid_updates_ref[i][j] = 0
    update_counts_ref = aggregation_result.get('update_counts')
    for i in range(global_params.num_centroids):
        update_counts_ref[i] = 0
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
    centroids_ref = global_state.get_ref('centroids')
    for i in range(global_params.num_centroids):
        for j in range(global_params.dims):
            centroids_ref[i][j] = 0
    global_state.iteration = 0
    global_state.done = False
    return global_state


def update_global_state(global_params, aggregation_result, global_state):
    """
    update global state object
    :global_params: global parameters object
    :aggregation_result: aggregation result
    :global_state: global state object to update
    :returns: global_state param
    """

    def dist(point1, point2):
        dist_sum = 0
        for i in range(global_params.dims):
            dist_sum += (point2 - point1) ** 2
        return math.sqrt(dist_sum)

    global_state.iteration += 1
    if global_state.iteration >= global_params.max_iteration:
        global_state.done = True
    agg_res_centroids_ref = aggregation_result.get_ref('centroid_updates')
    agg_res_counts_ref = aggregation_result.get_ref('update_counts')
    for i in range(global_params.num_centroids):
        if agg_res_counts_ref[i] == 0:
            continue
        for j in range(global_params.dims):
            agg_res_centroids_ref[i][j] /= agg_res_counts_ref[i]
    centroids_ref = global_state.get_ref('centroids')
    if all(dist(agg_res_centroids_ref[i], centroids_ref[i]) <=
           global_params.threshold for i in
           range(global_params.num_centroids)):
        global_state.done = True
    for i in range(global_params.num_centroids):
        if agg_res_counts_ref[i] == 0:
            continue
        for j in range(global_params.dims):
            centroids_ref[i][j] = agg_res_centroids_ref[i][j]
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
    _fields_ = [
        ('done', c_int),
        ('iteration', c_int),
        ('centroids', POINTER(POINTER(c_float))),
    ]
    aux_field_names = ('centroids',)

    def init_aux_structures(self, global_params):
        """
        initialize auxilary datastructures
        :global_params: global paramters object
        """
        self.centroids_aux = [
            [0] * global_params.dims for _ in
            range(global_params.num_centroids)
        ]

    def encode_dynamic(self, global_params):
        """
        encode dynamic alloced data that cannot be auto encoded
        :global_params: global parameters object
        :returns: b64 encoded json
        """
        if self.centroids:
            centroids = [
                [0] * global_params.dims for _ in
                range(global_params.num_centroids)
            ]
            for i in range(global_params.num_centroids):
                for j in range(global_params.dims):
                    centroids[i][j] = self.centroids[i][j]
        else:
            centroids = self.centroids_aux
        return user_prog_resources.b64_json_enc({'centroids': centroids})

    def decode_dynamic(self, encoded, global_params):
        """
        decode dynamic alloced data that cannot be auto encoded
        :global_params: global parameters object
        :encoded: b64 encoded json
        """
        decoded = user_prog_resources.b64_json_enc(encoded)
        if self.centroids:
            for i in range(global_params.num_centroids):
                for j in range(global_params.dims):
                    self.centroids[i][j] = decoded['centroids'][i][j]
        else:
            self.centroids_aux = decoded['centroids']


class AggregationResult(user_prog_resources.EncodableStructure):
    _fields_ = [
        ('centroid_updates', POINTER(POINTER(c_float))),
        ('update_counts', POINTER(c_int)),
    ]
    aux_field_names = ('centroid_updates', 'update_counts')

    def init_aux_structures(self, global_params):
        """
        initialize auxilary datastructures
        :global_params: global paramters object
        """
        self.centroid_updates_aux = [
            [0] * global_params.dims for _ in
            range(global_params.num_centroids)
        ]
        self.update_counts_aux = [0] * global_params.num_centroids

    def encode_dynamic(self, global_params):
        """
        encode dynamic alloced data that cannot be auto encoded
        :global_params: global parameters object
        :returns: b64 encoded json
        """
        if self.centroid_updates:
            centroid_updates = [
                [0] * global_params.dims for _ in
                range(global_params.num_centroids)
            ]
            for i in range(global_params.num_centroids):
                for j in range(global_params.dims):
                    centroid_updates[i][j] = self.centroid_updates[i][j]
            update_counts = [0] * global_params.num_centroids
            for i in range(global_params.num_centroids):
                update_counts[i] = self.update_counts[i]
        else:
            centroid_updates = self.centroid_updates_aux
            update_counts = self.update_counts_aux
        return user_prog_resources.b64_json_enc({
            'centroid_updates': centroid_updates,
            'update_counts': update_counts,
        })

    def decode_dynamic(self, encoded, global_params):
        """
        decode dynamic alloced data that cannot be auto encoded
        :global_params: global parameters object
        :encoded: b64 encoded json
        """
        raise NotImplementedError


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
            points = [[0] * global_params.dims for _ in range(self.num_points)]
            for i in range(self.num_points):
                for j in range(global_params.dims):
                    points[i][j] = self.points[i][j]
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
        ('num_centroids', c_int),
        ('threshold', c_float),
    ]
