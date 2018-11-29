import sys
from lizard import runtime_helper

runtime_helper.run_using_runner_module('config.user_programs.kmeans.runner',
        {'dims': 16, 'max_iterations': 10, 'input_file':
        'random-n2048-d16-c16.txt'}, 'http://localhost:5000',
        sys.argv[1])

