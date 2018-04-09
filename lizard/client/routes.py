from flask import Response
import json

from lizard.client import APP
from lizard import client, events

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
