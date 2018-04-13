import time
from lizard import LOG
from lizard import server


def shutdown_all_clients(max_wait=5, wait_interval=0.2):
    """
    shut down all clients because the server is terminating
    :max_wait: max time to wait for clients to shutdown before returning
    :wait_interval: poll interval to check if all clients have terminated
    """
    LOG.info('Instructing all clients to shutdown')
    with server.state_access() as s:
        s.get_all('/shutdown')
    for _ in range(int(max_wait / wait_interval)):
        time.sleep(wait_interval)
        with server.state_access() as s:
            client_count = len(s.clients)
        if client_count == 0:
            LOG.info('All clients terminated')
            break
    else:
        LOG.warn('Not all clients terminated, shutting down anyway')
