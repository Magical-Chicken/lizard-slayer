import functools
import logging
import sys
import threading
import queue

from lizard import LOG
from lizard import cli, client, cluster, hardware_discovery, server, util
from lizard.client import client_worker
from lizard.server import server_worker, server_util
from lizard import cuda


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
        server_util.shutdown_all_clients()
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
    import array
    a =  array.array('d', [1,2,3,6])
    for i in a: print(i)
    print(a)
    # cuda.test(a)
    res = cuda.aggregate(a, Dg=1, Db=32, Ns=0)
    print("res: ",  res)
    for i in a: print(i)
    print("kmeans")

    centers = array.array('d', [1,5])
    points =  array.array('d', [1,2,3,6])
    results = array.array('d', [0,0])
    cuda.kmeans_iteration(centers, points, results, k=2, dim=1, 
            Dg=1, Db=32, Ns=0)

    print("partial aggregations:")
    for i in results: print(i)
    
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
