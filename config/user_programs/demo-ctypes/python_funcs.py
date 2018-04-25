from ctypes import Structure, c_int, POINTER


class GlobalState(Structure):
    _fields_ = [
        ('iteration', c_int),
        ('values', POINTER(c_int)),
    ]


class AggregationResult(Structure):
    _fields_ = [
        ('values', POINTER(c_int)),
    ]


class Dataset(Structure):
    _fields_ = [
        ('points', POINTER(POINTER(c_int))),
    ]


class DatasetParams(Structure):
    _fields_ = [
        ('dims', c_int),
        ('num_points', c_int),
    ]
