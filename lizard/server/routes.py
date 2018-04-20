from flask import Response, request
import json

from lizard.server import APP
from lizard.server import remote_event, server_events
from lizard import server, events
from lizard import LOG

API_MIME_TYPE = 'application/json'


def respond_error(status=500):
    """
    Respond with a http error
    :status: http status code
    :returns: flask response
    """
    return Response("error: {}".format(status), status)


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
        event_type_name, server_events.ServerEventType)
    event = server_events.ServerEvent(e_type, data)
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
            client = state.clients.get(client_id)
        return (respond_json(client.properties) if client else
                respond_error(404))
    elif request.method == 'DELETE':
        with server.state_access() as state:
            res = state.clients.pop(client_id, None)
            LOG.info('Deleted client: %s', res)
        return Response("ok") if res is not None else respond_error(404)


@APP.route('/programs', methods=['GET', 'POST'])
def programs():
    """
    GET,POST /programs: register or list programs
    :returns: flask response
    """
    if request.method == 'POST':
        event_data = request.get_json()
        if not all(n in event_data for n in ('name', 'code', 'checksum')):
            return respond_error(400)
        return respond_create_event('register_prog', event_data)
    else:
        with server.state_access() as s:
            prog_hashes = list(s.registered_progs.keys())
        return respond_json({'programs': prog_hashes})


@APP.route('/programs/<prog_hash>', methods=['GET', 'DELETE'])
def program_item(prog_hash):
    """
    GET,DELETE /programs/<prog_hash>: query programs
    :prog_hash: program checksum/identifier
    :returns: flask response
    """
    if request.method == 'GET':
        with server.state_access() as s:
            prog = s.registered_programs.get(prog_hash)
        return respond_json(prog.properties) if prog else respond_error(404)
    else:
        raise NotImplementedError


@APP.route('/events/<event_id>', methods=['GET'])
def event_item(event_id):
    """
    GET /events/<event_id>: query event
    :event_id: event to return info for
    :returns: flask response
    """
    with server.SERVER_EVENT_MAP_LOCK:
        event = server.SERVER_EVENT_MAP.get(event_id)
    return respond_json(event.properties) if event else respond_error(404)


@APP.route('/remote_event/<client_id>/<event_id>', methods=['PUT'])
def remote_event_item(client_id, event_id):
    """
    PUT: /remote_event/<client_id>/<event_id>: update remote event
    :client_id: client id of remote event origin
    :event_id: event state to update
    :returns: flask response
    """
    event_props = request.get_json()
    with remote_event.remote_event_access() as r:
        r.store_event(client_id, event_id, event_props)
    return "ok"
