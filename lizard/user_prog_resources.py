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
    aux_field_names = []

    def encode(self):
        """encode as b64 encoded json string"""
        data = {}
        for field, _ in self._fields_:
            if field in self.aux_field_names:
                continue
            data[field] = getattr(self, field)
        if len(self.aux_field_names) > 0:
            data['aux'] = self.encode_aux()
        return b64_json_enc(data)

    def decode(self, encoded):
        """decode from b64 encoded json string"""
        data = b64_json_dec(encoded)
        for field, _ in self._fields_:
            if field in self.aux_field_names:
                continue
            setattr(self, field, data[field])
        if len(self.aux_field_names) > 0:
            self.decode_aux(data['aux'])

    def encode_aux(self):
        """encode auxilary data that cannot be auto encoded"""
        raise NotImplementedError

    def decode_aux(self, encoded):
        """decode auxiliry data that cannot be auto encoded"""
        raise NotImplementedError
