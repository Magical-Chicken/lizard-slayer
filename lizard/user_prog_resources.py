import base64
import ctypes
import json


def b64_json_enc(data):
    """
    encode data to b64 encoded json
    :data: data to encode
    :returns: encoded str
    """
    json_str = json.dumps(data)
    return base64.b64encode(json_str.encode()).decode()


def b64_json_dec(encoded):
    """
    decode data from b64 encoded json
    :encoded: encoded str
    :returns: data dict
    """
    json_str = base64.b64decode(encoded).decode()
    return json.loads(json_str)


class EncodableStructure(ctypes.Structure):
    """Base class for encodable ctypes structures"""
    _fields_ = []
    # NOTE: aux dynamic alloc fields must be non void pointer types
    aux_field_names = []

    def __len__(self):
        """
        len() *must* be defined for dataset objects, optional for others
        """
        pass

    def encode(self, global_params):
        """
        encode as b64 encoded json string
        :global_params: global parameters object
        :returns: b64 encoded json
        """
        data = {}
        for field, _ in self._fields_:
            if field in self.aux_field_names:
                continue
            data[field] = getattr(self, field)
        if len(self.aux_field_names) > 0:
            data['dynamic'] = self.encode_dynamic(global_params)
        return b64_json_enc(data)

    def decode(self, encoded, global_params, decode_aux=True):
        """
        decode from b64 encoded json string
        :global_params: global parameters object
        :encoded: b64 encoded json
        :decode_aux: if true decode aux fields
        """
        data = b64_json_dec(encoded)
        for field, _ in self._fields_:
            if field in self.aux_field_names:
                continue
            setattr(self, field, data[field])
        if decode_aux and len(self.aux_field_names) > 0:
            self.decode_dynamic(data['dynamic'], global_params)

    def get_ref(self, field):
        """
        get field reference, reverting to aux datastructures if no backing
        :field: field name string
        :returns: value of field
        :raises: AttributeError:
        """
        if field in self.aux_field_names and not getattr(self, field):
            field = '{}_aux'.format(field)
        return getattr(self, field)

    def init_aux_structures(self, global_params):
        """
        initialize auxilary datastructures
        :global_params: global paramters object
        """
        raise NotImplementedError

    def encode_dynamic(self, global_params):
        """
        encode dynamic alloced data that cannot be auto encoded
        :global_params: global parameters object
        :returns: b64 encoded json
        """
        raise NotImplementedError

    def decode_dynamic(self, encoded, global_params):
        """
        decode dynamic alloced data that cannot be auto encoded
        :global_params: global parameters object
        :encoded: b64 encoded json
        """
        raise NotImplementedError
