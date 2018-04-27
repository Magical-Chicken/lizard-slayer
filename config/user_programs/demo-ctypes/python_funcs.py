from ctypes import c_int, POINTER
from lizard import user_prog_resources


class GlobalState(user_prog_resources.EncodableStructure):
    _fields_ = [
        ('iteration', c_int),
        ('values', POINTER(c_int)),
    ]
    aux_field_names = ('values',)


class AggregationResult(user_prog_resources.EncodableStructure):
    _fields_ = [
        ('values', POINTER(c_int)),
    ]
    aux_field_names = ('values',)


class Dataset(user_prog_resources.EncodableStructure):
    _fields_ = [
        ('num_points', c_int),
        ('points', POINTER(POINTER(c_int))),
    ]
    aux_field_names = ('points',)


class GlobalParams(user_prog_resources.EncodableStructure):
    _fields_ = [
        ('dims', c_int),
        ('max_iterations', c_int),
    ]
