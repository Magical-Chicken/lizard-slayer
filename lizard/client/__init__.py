import contextlib
import flask
import threading

from lizard.client import lizard_client

APP = flask.Flask(__name__)
import lizard.client.routes  # NOQA

# global client state object
CLIENT = None
CLIENT_LOCK = threading.Lock()


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
