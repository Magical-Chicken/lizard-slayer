import enum
import time

from lizard import LOG
from lizard import util


class EventStatus(enum.Enum):
    PENDING = 'pending'
    RUNNING = 'running'
    SUCCESS = 'success'
    FAILURE = 'failure'


FINAL_STATES = (EventStatus.SUCCESS.value, EventStatus.FAILURE.value)


class BaseEvent(object):
    """Base event type"""
    event_map = None
    event_map_lock = None
    event_handler_map = None

    def __init__(self, event_type, data, register_event=True):
        """
        Init for Event
        :event_type: event type
        :data: event data
        :register_event: if true add event to event result map
        """
        self.event_type = event_type
        self.event_id = util.hex_uuid()
        self.status = EventStatus.PENDING
        self.result = None
        self.data = data
        self.completion_time = 0
        self._register_event()

    def handle(self):
        """
        Handle event using handler defined in event handler map and set result
        """
        if self.event_handler_map is None:
            raise NotImplementedError("Cannot handle BaseEvent")
        handler = self.event_handler_map.get(
            self.event_type, handler_not_implemented)
        start_time = time.time()
        try:
            self.status = EventStatus.RUNNING
            self.result = handler(self)
            self.status = EventStatus.SUCCESS
        except Exception as e:
            msg = repr(e)
            LOG.warning("Failed to complete event: %s error: %s", self, msg)
            self.status = EventStatus.FAILURE
            self.result = {'error': msg}
        end_time = time.time()
        self.completion_time = end_time - start_time

    def _register_event(self):
        """
        Register event in event map
        :result: result data
        """
        if self.event_map is None or self.event_map_lock is None:
            raise NotImplementedError("Cannot set result on BaseEvent")
        else:
            with self.event_map_lock:
                self.event_map[self.event_id] = self

    @property
    def properties(self):
<<<<<<< HEAD
        """event properties"""
=======
        """event properties, not including data which may be very large"""
        return {
            'event_id': self.event_id,
            'type': self.event_type.value,
            'status': self.status.value,
            'result': self.result,
            'completion_time': self.completion_time,
        }

    @property
    def full_properties(self):
        """full event properties including data"""
>>>>>>> ef9b13b186c1a356f50a36e78ad91a3ccff76392
        return {
            'event_id': self.event_id,
            'type': self.event_type.value,
            'status': self.status.value,
            'data': self.data,
            'result': self.result,
        }

    def __str__(self):
        """str repr for event"""
<<<<<<< HEAD
        return "Event: '{}' Data: {}".format(self.event_type, self.data)


class ClientEvent(BaseEvent):
    """Client event"""
    event_map = client.CLIENT_EVENT_MAP
    event_map_lock = client.CLIENT_EVENT_MAP_LOCK


class ServerEvent(BaseEvent):
    """Server event"""
    event_map = server.SERVER_EVENT_MAP
    event_map_lock = server.SERVER_EVENT_MAP_LOCK
=======
        return "EventType: '{}' ID: {}".format(self.event_type, self.event_id)
>>>>>>> ef9b13b186c1a356f50a36e78ad91a3ccff76392


def get_event_type_by_name(event_type_name, event_type_class):
    """
    get the event type object for the specified event name
    :event_type_name: event type name string
    :returns: instance of ClientEventType or ServerEventType
    """
    result = event_type_class.INVALID_TYPE
    if event_type_name in (e.value for e in event_type_class):
        result = event_type_class(event_type_name)
    return result


def handler_not_implemented(event):
    """
    placeholder event handler
    :event: event to handle
    :returns: event result data if event sucessfully handled
    :raises: Exception: if error occurs handling event
    """
    raise NotImplementedError("No event handler for event: {}".format(event))
