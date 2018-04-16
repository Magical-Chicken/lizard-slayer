from flask import Response, request
import json

from lizard.client import APP
from lizard import client, events
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
        event_type_name, events.ClientEventType)
    event = events.ClientEvent(e_type, data)
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
        post_data = request.get_json()
        prog_name = post_data['name']
        prog_code = post_data['code']
        prog_checksum = post_data['checksum']
        with client.client_access() as c:
            try:
                c.register_program(prog_name, prog_checksum, prog_code)
            except ValueError:
                return respond_error(400)
        return Response("ok")
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
    elif request.method == 'DELETE':
        with client.client_access() as c:
            res = c.user_programs.pop(prog_hash, None)
            LOG.info('Deleted user program: %s', res)
        return Response("ok") if res is not None else respond_error(404)


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
