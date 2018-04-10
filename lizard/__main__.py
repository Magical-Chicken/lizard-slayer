import functools
import logging
import sys
import threading
import queue

from lizard import LOG
from lizard import cli, client, cluster, hardware_discovery, server
from lizard.client import client_worker
from lizard.server import server_worker


def run_client(args):
    """
    entrypoint for client
    :args: parsed cmdline args
    :returns: 0 on success
    """
    # scan hardware
    hardware = hardware_discovery.scan_hardware(args)
    LOG.debug('hardware scan found: %s', hardware)
    # create client
    client.create_client(args, hardware)
    # start client api server
    # FIXME FIXME FIXME FIXME
    # automatically find available port
    call = functools.partial(
        client.APP.run, debug=False, host='0.0.0.0', port=6000)
    thread = threading.Thread(target=call)
    thread.daemon = True
    thread.start()
    # register with server
    with client.client_access() as c:
        c.register()
    # start client worker
    worker = client_worker.ClientWorker()
    LOG.info('starting client worker')
    try:
        worker.run()
    except queue.Empty:
        pass
    return 0


def run_server(args):
    """
    entrypoint for server
    :args: parsed cmdline args
    :returns: 0 on success
    """
    # create server state
    server.create_state(args)
    # start api server
    call = functools.partial(
        server.APP.run, debug=False, host=args.host, port=args.port)
    thread = threading.Thread(target=call)
    thread.daemon = True
    thread.start()
    # start server worker
    worker = server_worker.ServerWorker()
    LOG.info('starting server worker')
    try:
        worker.run()
    except queue.Empty:
        pass
    return 0


def run_cluster(args):
    """
    entrypoint for cluster
    :args: parsed cmdline args
    :returns: 0 on success
    """
    # create cluster state
    c = cluster.Cluster(args)
    # start cluster
    try:
        c.start()
    except KeyboardInterrupt:
        c.kill()
    return 0


def main():
    """
    main entry point
    :returns: 0 on success
    """
    subcmd_handlers = {
        'client': run_client,
        'server': run_server,
        'cluster': run_cluster,
    }

    # get the argument parser
    parser = cli.configure_parser()
    # parse arguments
    args = parser.parse_args()
    # normalize arguments
    args = cli.normalize_args(args)
    if args is None:
        return -1

    # set log level
    LOG.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    LOG.debug('logging system init')
    LOG.debug('running with args: %s', args)

    # exit success
    return subcmd_handlers[args.subcmd](args)


if __name__ == "__main__":
    sys.exit(main())
