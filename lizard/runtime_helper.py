import os
import requests
import time

from lizard.events import FINAL_STATES


def poll_for_event_complete(
        server_address, event_id, poll_freq=1, max_time=600):
    """
    poll for an event to complete
    :server_address: full uri for server
    :event_id: event uuid
    :poll_freq: polling frequency (seconds)
    :max_time: max wait time (seconds)
    :returns: event data
    """
    for _ in range(int(max_time / poll_freq)):
        path = os.path.join(server_address, 'events', event_id)
        data = requests.get(path).json()
        if data['status'] in FINAL_STATES:
            break
        time.sleep(poll_freq)
    return data
