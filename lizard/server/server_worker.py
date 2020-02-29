from lizard import LOG
from lizard.server import SERVER_QUEUE
from lizard.server.server_events import ServerEventType


class ServerWorker(object):
    """Main server worker object"""

    def __init__(self):
        """
        ServerWorker init
        """
        self.shutdown_scheduled = False

    def handle_event(self, event):
        """
        handle an event
        :event: event to handle
        """
        if event.event_type == ServerEventType.REQ_SHUTDOWN:
            self.shutdown_scheduled = True
        else:
            event.handle()

    def run(self):
        """
        main loop, wait for event, then process
        if shutdown scheduled continue until queue empty
        :returns: does not return, uses exception to end control
        :raises queue.Empty: when shutdown requested and queue empty
        """
        while True:
            event = SERVER_QUEUE.get(block=not self.shutdown_scheduled)
            LOG.debug('received event: %s', event)
            self.handle_event(event)
            LOG.debug('handled event: %s', event)
            SERVER_QUEUE.task_done()
