import functools
import logging
import sys
import threading
import queue

from lizard import LOG
from lizard import cli, client, cluster, hardware_discovery, server, util
from lizard.client import client_worker
from lizard.server import remote_event, server_worker, server_util


def run_client(args, tmpdir):
    """
    entrypoint for client
    :args: parsed cmdline args
    :tmpdir: temporary directory
    :returns: 0 on success
    """
    # scan hardware
    hardware = hardware_discovery.scan_hardware(args, tmpdir)
    # create client
    client.create_client(args, tmpdir, hardware)
    # automatically find available port
    client_port = util.get_free_port()
    # start client api server
    call = functools.partial(
        client.APP.run, debug=False, host='0.0.0.0', port=client_port)
    thread = threading.Thread(target=call)
    thread.daemon = True
    thread.start()
    # register with server
    with client.client_access() as c:
        c.register(client_port)
    # start client worker
    worker = client_worker.ClientWorker()
    LOG.info('starting client worker')
    try:
        worker.run()
    except queue.Empty:
        with client.client_access() as c:
            c.shutdown()
    return 0


def run_server(args, tmpdir):
    """
    entrypoint for server
    :args: parsed cmdline args
    :tmpdir: temporary directory
    :returns: 0 on success
    """
    # create server state
    server.create_state(args, tmpdir)
    # init remote event system
    remote_event.create_remote_events()
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
        server_util.shutdown_all_clients()
    return 0


def run_cluster(args, tmpdir):
    """
    entrypoint for cluster
    :args: parsed cmdline args
    :tmpdir: temporary directory
    :returns: 0 on success
    """
    # create cluster state
    c = cluster.Cluster(args, tmpdir)
    # start cluster
    try:
        c.start()
    except KeyboardInterrupt:
        # FIXME will kill all python processes including server
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

    # create tmpdir and run handler
    with util.TempDir(preserve=args.keep_tmpdir) as tmpdir:
        return subcmd_handlers[args.subcmd](args, tmpdir)


if __name__ == "__main__":
    sys.exit(main())
