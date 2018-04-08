from lizard.client import APP


@APP.route('/ruok')
def ruok():
    """
    GET /ruok: check if server is running, expect response 'imok'
    :returns: flask response
    """
    return 'imok'
