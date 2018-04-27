from flask import Response, request
import json

from lizard.client import APP
from lizard.client import client_events
from lizard import client, events

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


def respond_create_event(event_type_name, data, send_remote_event=False):
    """
    create event and add to queue
    :event_type: even ttype name
    :data: event details
    :send_remote_event: if true send event status to server
    :returns: flask response
    """
    e_type = events.get_event_type_by_name(
        event_type_name, client_events.ClientEventType)
    event = client_events.ClientEvent(
        e_type, data, send_remote_event=send_remote_event)
    client.CLIENT_QUEUE.put_nowait(event)
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


@APP.route('/programs', methods=['GET', 'POST'])
def programs():
    """
    GET,POST /programs: register or list programs
    :returns: flask response
    """
    if request.method == 'POST':
        event_data = request.get_json()
        if not all(n in event_data for n in ('name', 'data', 'checksum')):
            return respond_error(400)
        return respond_create_event(
            'register_prog', event_data,
            send_remote_event=event_data.get('send_remote_event'))
    else:
        with client.client_access() as c:
            prog_hashes = list(c.user_programs.keys())
        return respond_json({'programs': prog_hashes})


@APP.route('/programs/<prog_hash>', methods=['GET', 'DELETE'])
def program_item(prog_hash):
    """
    GET,DELETE /programs/<prog_hash>: query programs
    :prog_hash: program checksum/identifier
    :returns: flask response
    """
    if request.method == 'GET':
        with client.client_access() as c:
            prog = c.user_programs.get(prog_hash)
        return respond_json(prog.properties) if prog else respond_error(404)
    else:
        raise NotImplementedError


@APP.route('/runtimes/<prog_hash>/<runtime_id>', methods=['POST'])
def runtime_init(prog_hash, runtime_id):
    """
    POST /runtimes/<prog_hash>/<runtime_id>: init program runtime
    :prog_hash: program checksum/identifier
    :runtime_id: runtime uuid
    :returns: flask response
    """
    data = request.get_json()
    event_data = {
        'checksum': prog_hash,
        'runtime_id': runtime_id,
        'dataset_enc': data['dataset_enc'],
        'global_params_enc': data['global_params_enc'],
        'data_count': data['data_count'],
    }
    return respond_create_event(
        'init_runtime', event_data,
        send_remote_event=data.get('send_remote_event'))


@APP.route('/runtimes/<prog_hash>/<runtime_id>/iterate', methods=['POST'])
def run_iteration(prog_hash, runtime_id):
    """
    POST /runtimes/<prog_hash>/<runtime_id>/iterate: run iteration
    :prog_hash: program checksum/identifier
    :runtime_id: runtime uuid
    :returns: flask response
    """
    data = request.get_json()
    event_data = {
        'checksum': prog_hash,
        'runtime_id': runtime_id,
        'global_state_enc': data['global_state_enc'],
    }
    return respond_create_event(
        'run_iteration', event_data,
        send_remote_event=data.get('send_remote_event'))


@APP.route('/events/<event_id>', methods=['GET'])
def event_item(event_id):
    """
    GET /events/<event_id>: query event
    :event_id: event to return info for
    :returns: flask response
    """
    with client.CLIENT_EVENT_MAP_LOCK:
        event = client.CLIENT_EVENT_MAP.get(event_id)
    return respond_json(event.properties) if event else respond_error(404)
