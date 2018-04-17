from lizard import LOG
from lizard.client import CLIENT_QUEUE
from lizard.client.client_events import ClientEventType


class ClientWorker(object):
    """Main worker object"""

    def __init__(self):
        """
        ClientWorker init
        """
        self.shutdown_scheduled = False

    def handle_event(self, event):
        """
        handle an event
        :event: event to handle
        """
        if event.event_type == ClientEventType.REQ_SHUTDOWN:
            self.shutdown_scheduled = True
        else:
            raise NotImplementedError

    def run(self):
        """
        main loop, wait for event, then process
        if shutdown scheduled continue until queue empty
        :returns: does not return, uses exception to end control
        :raises queue.Empty: when shutdown requested and queue empty
        """
        while True:
            event = CLIENT_QUEUE.get(block=not self.shutdown_scheduled)
            LOG.debug('received event: %s', event)
            self.handle_event(event)
            LOG.debug('handled event: %s', event)
            CLIENT_QUEUE.task_done()
