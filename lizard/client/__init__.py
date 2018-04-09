import contextlib
import flask
import threading
import queue

from lizard.client import lizard_client

# FIXME FIXME FIXME
# using flask on the client is temporary for dev purposes only
APP = flask.Flask(__name__)

# global client state object
CLIENT = None
CLIENT_LOCK = threading.Lock()

# global client event map
CLIENT_EVENT_MAP = {}
CLIENT_EVENT_MAP_LOCK = threading.Lock()

# global client task queue
CLIENT_QUEUE = queue.Queue()


def create_client(args, hardware):
    """
    Create client object
    :args: parsed cmdline rags
    :hardware: hardware info dict
    """
    with CLIENT_LOCK:
        global CLIENT
        CLIENT = lizard_client.LizardClient(args, hardware)


@contextlib.contextmanager
def client_access():
    """contextmanager to provide access to global client"""
    with CLIENT_LOCK:
        yield CLIENT


# import routes
import lizard.client.routes  # NOQA
