import contextlib
import threading

import flask
APP = flask.Flask(__name__)
import lizard.server.routes  # NOQA

# global server state object
SERVER_STATE = None
SERVER_STATE_LOCK = threading.Lock()


@contextlib.contextmanager
def state_access():
    """contextmanager to provide access to global server state"""
    with SERVER_STATE_LOCK:
        yield SERVER_STATE
