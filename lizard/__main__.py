import logging
import sys

from lizard import LOG
from lizard import cli, client, hardware_discovery, server


def run_client(args):
    """
    entrypoint for client
    :args: parsed cmdline args
    :returns: 0 on success
    """
    # scan hardware
    hardware = hardware_discovery.scan_hardware()
    LOG.debug('hardware scan found: %s', hardware)
    lizard_client = client.LizardClient(args, hardware)
    lizard_client.register()
    return 0


def run_server(args):
    """
    entrypoint for server
    :args: parsed cmdline args
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
