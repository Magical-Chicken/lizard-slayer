#!/usr/bin/python3

# FIXME FIXME FIXME
# When an api is written to streamline using this platform this can be removed
# This is brittle, has hardcoded paths, and is not intended for production use

# Usage:
#   load_and_run.py program/dir/ http://server_address:port_number

import hashlib
import json
import os
import requests
import sys
import yaml

CONFIG_FILENAME = 'config.yaml'


def load_program(program_directory, config_file):
    with open(config_file, 'r') as fp:
        config_data = yaml.load(fp)
    for key, source_name in config_data['sources'].items():
        source_path = os.path.join(program_directory, source_name)
        with open(source_path, 'r') as fp:
            config_data['code'][key] = fp.read()
    return config_data


def register_program(server_address, config_data):
    config_json = json.dumps(config_data)
    checksum = hashlib.sha1(config_json.encode()).hexdigest()
    post_data = {
        'name': config_data['info']['name'],
        'data': config_json,
        'checksum': checksum,
    }
    endpoint = os.path.join(server_address, 'programs')
    requests.post(endpoint, json=post_data)
    # TODO: block until program ready, requires blocking api endpoint support
    return checksum


def main():
    program_directory = sys.argv[1]
    server_address = sys.argv[2]
    config_file = os.path.join(program_directory, CONFIG_FILENAME)
    if not os.path.isdir(program_directory) or not os.path.exists:
        print("Bad program directory: {}".format(program_directory))
        return -1
    config_data = load_program(program_directory, config_file)
    checksum = register_program(server_address, config_data)
    print("Program registered, checksum: {}".format(checksum))
    return 0


if __name__ == "__main__":
    sys.exit(main())
