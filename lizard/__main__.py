import logging
import sys

from lizard import LOG
from lizard import cli, server


def run_client(args):
    """
    entrypoint for client
    :returns: 0 on success
    """
    raise NotImplementedError
    return 0


def run_server(args):
    """
    entrypoint for server
    :returns: 0 on success
    """
    server.APP.run(host=args.host, port=args.port, debug=False)
    return 0


def main():
    """
    main entry point
    :returns: 0 on success
    """
    subcmd_handlers = {'client': run_client, 'server': run_server}

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
