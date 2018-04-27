from ctypes import c_int, POINTER
from lizard import user_prog_resources


class GlobalState(user_prog_resources.EncodableStructure):
    _fields_ = [
        ('iteration', c_int),
        ('values', POINTER(c_int)),
    ]
    aux_field_names = ('values',)

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
        ('values', POINTER(c_int)),
    ]
    aux_field_names = ('values',)

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
        ('points', POINTER(POINTER(c_int))),
    ]
    aux_field_names = ('points',)

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
        if self.values:
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
