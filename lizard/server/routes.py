from lizard.server import APP

API_MIME_TYPE = 'application/json'


@APP.route('/ruok')
def ruok():
    """
    GET /ruok: check if server is running, expect response 'imok'
    :returns: flask response
    """
    return 'imok'


@APP.route('/register')
def register():
    """
    GET /register: register a client with the server
    :returns: flask response
    """
    raise NotImplementedError
