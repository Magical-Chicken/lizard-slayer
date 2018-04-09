import contextlib
import flask
import threading
import queue

from lizard.server import state

APP = flask.Flask(__name__)
import lizard.server.routes  # NOQA

# global server state object
SERVER_STATE = None
SERVER_STATE_LOCK = threading.Lock()

# global server event map
SERVER_EVENT_MAP = {}
SERVER_EVENT_MAP_LOCK = threading.Lock()

# global server task queue
SERVER_QUEUE = queue.Queue()


def create_state(args):
    """
    Create server state object
    :args: parsed cmdline args
    """
    with SERVER_STATE_LOCK:
        global SERVER_STATE
        SERVER_STATE = state.ServerState(args)


@contextlib.contextmanager
def state_access():
    """contextmanager to provide access to global server state"""
    with SERVER_STATE_LOCK:
        yield SERVER_STATE
