from ctypes import c_int, POINTER
import resources


class GlobalState(resources.EncodableStructure):
    _fields_ = [
        ('iteration', c_int),
        ('values', POINTER(c_int)),
    ]
    aux_field_names = ('values',)


class AggregationResult(resources.EncodableStructure):
    _fields_ = [
        ('values', POINTER(c_int)),
    ]
    aux_field_names = ('values',)


class Dataset(resources.EncodableStructure):
    _fields_ = [
        ('num_points', c_int),
        ('points', POINTER(POINTER(c_int))),
    ]
    aux_field_names = ('points',)


class GlobalParams(resources.EncodableStructure):
    _fields_ = [
        ('dims', c_int),
        ('max_iterations', c_int),
    ]
