from flask import Response, request
import json

from lizard.server import APP
from lizard import server, events
from lizard import LOG

API_MIME_TYPE = 'application/json'


def respond_json(data, status=200):
    """
    Respond to a request with a json blob
    :data: dict of data
    :status: http status code
    :returns: flask response
    """
    return Response(json.dumps(data), status, mimetype=API_MIME_TYPE)


def respond_create_event(event_type_name, data):
    """
    create event and add to queue
    :event_type: even ttype name
    :data: event details
    :returns: flask response
    """
    e_type = events.get_event_type_by_name(
        event_type_name, events.ServerEventType)
    event = events.ServerEvent(e_type, data)
    server.SERVER_QUEUE.put_nowait(event)
    return respond_json({'event_id': event.event_id})


@APP.route('/ruok')
def ruok():
    """
    GET /ruok: check if server is running, expect response 'imok'
    :returns: flask response
    """
    return 'imok'


@APP.route('/shutdown')
def shutdown():
    """
    GET /shutdown: schedule client shutdown
    :returns: flask response
    """
    return respond_create_event('req_shutdown', {})


@APP.route('/clients', methods=['GET', 'POST'])
def clients():
    """
    GET,POST /instances/: register or list clients
    :returns: flask response
    """
    if request.method == 'POST':
        post_data = request.get_json()
        client_hardware = post_data['hardware']
        client_port = post_data['client_port']
        client_ip = request.remote_addr
        with server.state_access() as state:
            client_uuid = state.register_client(
                client_hardware, client_ip, client_port)
        return respond_json({'uuid': client_uuid})
    else:
        with server.state_access() as state:
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
        with server.state_access() as state:
            client = state.clients[client_id]
            client_data = client.properties if client is not None else {}
        return respond_json(client_data, status=200 if client_data else 404)
    elif request.method == 'DELETE':
        with server.state_access() as state:
            res = state.clients.pop(client_id, None)
            LOG.info('Deleted client: %s', res)
        return Response("ok") if res is not None else Response("bad id", 404)
