from flask import Response, request
import json

from lizard.server import APP
from lizard.server import state as server_state

API_MIME_TYPE = 'application/json'


def respond_json(data, status=200):
    """
    Respond to a request with a json blob
    :data: dict of data
    :status: http status code
    :returns: flask response
    """
    return Response(json.dumps(data), status, mimetype=API_MIME_TYPE)


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
        client_hardware = request.get_json()
        with server_state.state_access() as state:
            client_uuid = state.register_client(client_hardware)
        return respond_json({'uuid': client_uuid})
    else:
        with server_state.state_access() as state:
            client_uuids = list(state.clients.keys())
        return respond_json({'clients': client_uuids})


@APP.route('/clients/<client_id>', methods=['GET', 'DELETE'])
def client_item(client_id):
    """
    GET,DELETE /clients/<client_id>: query clients
    :client_id: client uuid
    :returns: flask response
    """
    if request.method == 'GET':
        with server_state.state_access() as state:
            client = state.clients[client_id]
            client_data = client.properties if client is not None else {}
        return respond_json(client_data, status=200 if client_data else 404)
    elif request.method == 'DELETE':
        with server_state.state_access() as state:
            res = state.clients.pop(client_id, None)
        return Response("ok") if res is not None else Response("bad id", 404)
