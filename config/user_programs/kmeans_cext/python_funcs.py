import re
import array
import random
import math
from itertools import chain

# import user_program

# FIXME very bad code for userprog

def split_iter(string):
    return (x.group(0) for x in re.finditer(r"[A-Za-z']+", string))


def test():
    return 'test func'


def data_header(data):

    # get first line
    index = data.find('\n')
    # split count and dim
    return list(map(int,data[:index].split()))


def split_data(data):

    # count = int(data[0].strip())
    # it = split_data(data)
    # skip first line
    # next(it)
    d = data.split('\n')

    # file layout:
    # count dim k
    # 1 d1 d2 d3 .....\n
    # 2 d1 d2 d3 .....\n
    # 3 d1 d2 d3 .....\n
    ...
    generator = (list(map(float, line.split()[1:])) for line in d[1:-1])

    return generator


def initialize_global_state(params):
    dim = params[1] # dim
    k = params[2] # k
    # centers = array.array('d', (random.random() for _ in range(k * dim)))
    centers = [random.random() for _ in range(k * dim)]
    return [0, centers]


def to_array(centers):
    
    a = array.array('d', chain.from_iterable(centers))
    return a
     

# FIXME handle unpinned memory
def run_iteration(params, data_count, global_state, pinned_memory, dataset):
    import user_program
    # count = params[0]
    dim = params[1] # dim
    k = params[2] # k
    count_results = array.array('i', [0]*k)
    partial_results = array.array('d', [0]*k*dim)

        
    user_program.kmeans_iteration(global_state, pinned_memory, partial_results,
            count_results, data_count, k, dim, data_count // 64 + 1, 64, 0)

    for i in range(k):
        if count_results[i]==0:
            r = random.randrange(0,data_count)
            count_results[i] = 1
            for d in range(dim):
                partial_results[i*dim+d] = dataset[i*dim +d]

    count = []
    partial = []

    for c in count_results:
        count.append(c)
    for p in partial_results:
        partial.append(p)
    return [count, partial]
    #return partial_results


def init_aggregation_result(params, aggregation_result=None):
    dim = params[1] # dim
    k = params[2] # k
    aggregation_result = [[0]*k, [0]*k*dim]
    return aggregation_result


def aggregate(params, aggregation_result, partial_result):
    dim = params[1] # dim
    k = params[2] # k
    for i in range(k):
        aggregation_result[0][i] = aggregation_result[0][i] + partial_result[0][i]

    for i in range(k*dim):
        aggregation_result[1][i] = aggregation_result[1][i] + partial_result[1][i]


def update_global_state(params, aggregation_result, global_state):
    dim = params[1] # dim
    k = params[2] # k

    centers = global_state[1]

    for i in range(k):
        if aggregation_result[0][i] == 0:
            for d in range(dim):
                centers[i*dim + d] = random.random()
        else:
            for d in range(dim):
                centers[i*dim + d] = aggregation_result[1][i*dim + d] / aggregation_result[0][i]
    global_state[0] = global_state[0] + 1

    return global_state


def terminate(params, prev_global_state, global_state):
    dim = params[1] # dim
    k = params[2] # k
    max_iterations = params[3] # k
    threshold = params[4] # k

    current_iteration = global_state[0]
    centers = global_state[1]

    if current_iteration > max_iterations:
        return True


    for i in range(k):
        mag = 0
        for d in range(dim):
            difference = prev_global_state[1][i*dim + d] - centers[i*dim + d]
            mag = mag + difference * difference
        
        distance = math.sqrt(mag)
        if distance > threshold:
            return False

    return True
