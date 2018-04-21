import enum
import os

from lizard import LOG
from lizard import client, events, user_prog


class ClientEventType(enum.Enum):
    INVALID_TYPE = 'invalid_type'
    REQ_SHUTDOWN = 'req_shutdown'
    REGISTER_PROG = 'register_prog'
    REQ_RUN_PROG = 'req_run_prog'


def handle_event_register_prog(event):
    """
    handle 'register_prog' event
    data must include 'name', 'checksum' and 'data'
    :event: event to handle
    :returns: event result data if event sucessfully handled
    :raises: ValueError: if program data does not match checksum
    """
    data = event.data['data']
    name = event.data['name']
    checksum = event.data['checksum']
    with client.client_access() as c:
        user_progs_dir = c.user_progs_dir
    prog_dir = os.path.join(user_progs_dir, checksum)
    data_file = os.path.join(prog_dir, 'data.json')
    os.mkdir(prog_dir)
    with open(data_file, 'w') as fp:
        fp.write(data)
    program = user_prog.UserProg(name, checksum, data_file, build_dir=prog_dir)
    program.verify_checksum()
    program.build()
    with client.client_access() as c:
        c.user_programs[checksum] = program
    LOG.info('Registered program: %s', program)
    return {}


CLIENT_EVENT_HANDLER_MAP = {
    ClientEventType.REGISTER_PROG: handle_event_register_prog,
}


class ClientEvent(events.BaseEvent):
    """Client event"""
    event_map = client.CLIENT_EVENT_MAP
    event_map_lock = client.CLIENT_EVENT_MAP_LOCK
    event_handler_map = CLIENT_EVENT_HANDLER_MAP

    def __init__(
            self, event_type, data, register_event=True,
            send_remote_event=False):
        """
        Init for Event
        :event_type: event type
        :data: event data
        :register_event: if true add event to event result map
        :send_remote_event: if true send event status to server
        """
        self.send_remote_event = send_remote_event
        super().__init__(event_type, data, register_event=register_event)

    def handle(self):
        """
        Handle event using handler defined in event handler map and set result
        """
        super().handle()
        if self.send_remote_event:
            self._put_remove_event()

    def _register_event(self):
        """
        Register event in event map
        :result: result data
        """
        super()._register_event()
        if self.send_remote_event:
            self._put_remove_event()

    def _put_remove_event(self):
        """PUT event status in server remote event map"""
        with client.client_access() as c:
            endpoint = os.path.join('/remote_event', c.uuid, self.event_id)
            c.put(endpoint, self.properties, expect_json=False, add_uuid=False)
