import contextlib
import threading
import queue

from lizard.events import EventStatus

REMOTE_EVENTS = None
REMOTE_EVENTS_LOCK = threading.Lock()


def create_remote_events():
    """Create remote events object"""
    with REMOTE_EVENTS_LOCK:
        global REMOTE_EVENTS
        REMOTE_EVENTS = RemoteEvents()


@contextlib.contextmanager
def remote_events_access():
    """contextmanager to provide access to remote events"""
    with REMOTE_EVENTS_LOCK:
        yield REMOTE_EVENTS


class RemoteEvents(object):
    """Manages tracking remote events"""
    final_states = (EventStatus.SUCCESS.value, EventStatus.FAILURE.value)

    def __init__(self):
        """RemoteEvents init"""
        self._remote_event_map = {}
        self._callback_map = {}
        self._multi_callback_list = []
        self._callback_queue = queue.Queue()
        self._complete_event_set = set()
        self.start_callback_worker()

    def register_client(self, client_id):
        """
        Set up tracking for new client
        :client_id: client uuid
        :raises: ValueError: if client already registerd
        """
        if client_id in self._remote_event_map:
            raise ValueError("Client already exists")
        self._remote_event_map[client_id] = {}
        self._callback_map[client_id] = {}

    def register_callback_func(self, client_id, event_id, func):
        """
        Register a callback function
        :client_id: client uuid
        :event_id: event uuid
        :func: function call to register, must accept event data as only arg
        """
        if event_id not in self._callback_map[client_id]:
            self._callback_map[client_id][event_id] = []
        self._callback_map[client_id][event_id].append(func)

    def register_multi_callback(self, event_ids, callback_func):
        """
        Register a callback to run after a group of event_ids complete
        :event_ids: group of event ids to wait for
        :callback_func: callback function to run
        """
        # NOTE: this assumes client generated uuids are globally unique
        self._multi_callback_list.append({
            'funcall': callback_func,
            'event_ids': event_ids
        })
        for event_id in self._complete_event_set.intersection(set(event_ids)):
            self.remove_from_multi_callback_pending(event_id)

    def run_callbacks_for_event(self, client_id, event_id, event_props):
        """
        run all callback functions registered for event
        :client_id: client uuid
        :event_id: event uuid
        :event_props: event properties
        """
        funcs = self._callback_map[client_id].pop(event_id, [])
        for func in funcs:
            callback_data = {'funcall': func, 'event_data': event_props}
            self._callback_queue.put_nowait(callback_data)

    def remove_from_multi_callback_pending(self, event_id):
        """
        remove the event from the callback pendings for all multi callbacks
        :event_id: event uuid
        :event_props: event properties
        """
        for multi_info in self._multi_callback_list:
            if event_id in multi_info['event_ids']:
                multi_info['event_ids'].remove(event_id)
            if len(multi_info['event_ids']) == 0:
                callback_data = {
                    'funcall': multi_info['funcall'],
                    'event_data': {}
                }
                self._callback_queue.put_nowait(callback_data)

    def store_event(self, client_id, event_id, event_props):
        """
        Update or register an event
        :client_id: client uuid
        :event_id: event uuid
        :event_props: new event properties
        :raises: ValueError: if client or event does not exist
        """
        if client_id not in self._remote_event_map:
            raise ValueError("Client does not exist")
        self._remote_event_map[client_id][event_id] = event_props
        if event_props['status'] in self.final_states:
            self.run_callbacks_for_event(client_id, event_id, event_props)
            self.remove_from_multi_callback_pending(event_id)
            self._complete_event_set.add(event_id)

    def start_callback_worker(self):
        """Worker for handling remote event triggered callbacks"""

        def callback_worker():
            while True:
                callback_data = self._callback_queue.get(block=True)
                callback_data['funcall'](callback_data['event_data'])
                self._callback_queue.task_done()

        self._callback_worker_thread = threading.Thread(target=callback_worker)
        self._callback_worker_thread.daemon = True
        self._callback_worker_thread.start()
