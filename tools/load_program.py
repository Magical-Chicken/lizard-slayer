#!/usr/bin/python3

# Usage:
#   load_program.py program/dir/ http://server_address:port_number

import hashlib
import json
import os
import requests
import sys
import yaml

try:
    from lizard import runtime_helper
except ImportError:
    parent_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    sys.path.append(os.path.abspath(parent_dir))
    from lizard import runtime_helper

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
    req_data = requests.post(endpoint, json=post_data).json()
    event_id = req_data['event_id']
    runtime_helper.poll_for_event_complete(server_address, event_id)
    return checksum

def run_program(server_address, checksum, data_file):
    endpoint = os.path.join(server_address, 'run', checksum)
    with open(data_file, 'r') as fp:
        data = fp.read()
    post_data = {
        'checksum': checksum,
        'data': data,
    }
    requests.post(endpoint, json=post_data)

def main():
    program_directory = sys.argv[1]
    server_address = sys.argv[2]
    data_file = sys.argv[3]
    config_file = os.path.join(program_directory, CONFIG_FILENAME)
    if not os.path.isdir(program_directory) or not os.path.exists:
        print("Bad program directory: {}".format(program_directory))
        return -1
    config_data = load_program(program_directory, config_file)
    checksum = register_program(server_address, config_data)
    print("Program registered, checksum: {}".format(checksum))

    print("Running program")
    run_program(server_address, checksum, data_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
