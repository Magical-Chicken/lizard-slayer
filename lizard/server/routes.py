from flask import Response, request
import json

from lizard.server import APP
from lizard.server import state as server_state

API_MIME_TYPE = 'application/json'


def respond_json(data):
    """
    Respond to a request with a json blob
    """
    return Response(json.dumps(data), mimetype=API_MIME_TYPE)


@APP.route('/ruok')
def ruok():
    """
    GET /ruok: check if server is running, expect response 'imok'
    :returns: flask response
    """
    return 'imok'


@APP.route('/clients', methods=['GET', 'POST'])
def clients():
    """
    GET,POST /instances/: register or list clients
    :returns: flask response
    """
    if request.method == 'POST':
        client_hardware = request.form['hardware']
        with server_state.state_access() as state:
            client_uuid = state.register_client(client_hardware)
        return respond_json({'uuid': client_uuid})
    else:
        raise NotImplementedError
