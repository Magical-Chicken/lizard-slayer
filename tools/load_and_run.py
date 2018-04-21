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
EXPECTED_SOURCE_FILES_MAP = {
    'python_source': 'python_code',
    'cpp_source': 'cpp_code',
    'cuda_source': 'cuda_code',
}


def load_program(program_directory, config_file):
    with open(config_file, 'r') as fp:
        config_data = yaml.load(fp)
    for source, config_key_name in EXPECTED_SOURCE_FILES_MAP.items():
        source_name = config_data['sources'][source]
        source_path = os.path.join(program_directory, source_name)
        with open(source_path, 'r') as fp:
            config_data['code'][config_key_name] = fp.read()
    return config_data


def register_program(server_address, config_data):
    config_json = json.dumps(config_data)
    checksum = hashlib.sha1(config_json).hexdigest()
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
    program_directory = sys.argv[0]
    server_address = sys.argv[1]
    config_file = os.path.join(program_directory, CONFIG_FILENAME)
    if not os.path.isdir(program_directory) or not os.path.exists:
        print("Bad program directory")
        return -1
    config_data = load_program(program_directory, config_file)
    checksum = register_program(server_address, config_data)
    print("Program registered, checksum: {}".format(checksum))
    return 0


if __name__ == "__main__":
    sys.exit(main())
