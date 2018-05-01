# lizard-slayer
A platform for running CUDA programs across a cluster.

## Program definitions
Programs definitions consist of a CUDA source file, a C++ header with C style
structs for defining program data objects, a python module with python
interfaces to the C structs used, and a config file to be read by the program
loader script.

### CUDA/C++ components
All user programs must use 4 structures, defined in C++ with corresponding
python definitions using ctypes.Structure. These include:
 - A global parameters structure, with user provided run configuration
 - A dataset structure, that holds user provided data
 - A aggregation result structure, with the result of an iteration or
   aggregation
 - A global state structure, with final result information and any information
   that must be available at each iteration

All user programs must implement methods to allocate and deallocate these
structure, which must be named correctly and have the correct signiture, see
the demo_ctypes program header for an example. In addition, they must implement
a run iteration method, again see the demo_ctypes program header for the
method's signature.

### Python components
The ctypes.Structure wrappers defined in the python module must, in addition to
listing the fields of the C style structs, specify which of those fields are
dynamically allocated and provide functions to manage encoding data stored in
dynamically allocated memory and to configure pure python temporary storage for
manipulating data from the dynamically allocated fields entirely within python.
See the demo_ctypes program's python_funcs module for an example.

The python module for a user program must also provide the following methods:
 - A pure python implementation of the associative aggregation function,
   matching the aggregation phase of the algorithm being used in the user
   program
 - A function to initialize an aggregation result
 - A function to initialize a global state object
 - A function to update a global state object with an aggregation result,
   and set the program completed variable if no further iterations are needed
 - A function to get a slice of a dataset object

### Configuration file
The configuration file must specify the names of the source files for the user
program, and may provide some metadata about the program, such as a name and
verison number. The user program must also specify which type of interface to
use between it and the python system. This is set using the `py_c_extention`
field in the info section of the config file. If true, the program is treated
as a python extension module, if false, ctypes is used to interact with the
program. The `py_c_extention` field must be set to false, as only ctypes is
currently supported. See the demo_ctypes program's config file for an example.

### Optinal runner module
In addition to the main program definition, a module may be specified which
provides functions to streamline running a user program, using the
`runtime_helper.run_using_runner_module` function. The runner module must
provide a function to initialize global parameters from user provided runtime
settings, and to initialize a dataset from user provided runtime settings. In
addition, it must include a function to print out the result of a program run.
See the runner module in the demo_ctypes program for an example.

## Running the platform
The platform should always be started via the `tox` virtual environment
manager, as it will automatically handle python dependancies. A tox environment
called `run` is provided, wich can be used with the command
`tox -e run -- <args>`. To run without tox, install all dependancies listed in
the program requirements and ensure that version numbers are correct.

When running the platform, the subcommand must be specified, then the arguments
for that subcommand. Run with the `--help` flag for more information. It is
recommended to run every command with the `-v` flag, which enables debug
messages.

### Running the server
The server should always be started before the clients, as the clients will
fail if they cannot connect to the server. To run the server, no additional
arguments need be specified.

For example `tox -e run -- server -v`

### Running the client
Once the server is running, the clients can be started on each worker node. The
clients must be told the hostname/ip address of the server, the port the server
is accepting connections on (defaults to 5000), and optionally the path to the
directory the CUDA compiler is in and the path to the CUDA lib64.

For example `tox -e run -- client -v -a localhost -p 5000 -b /opt/cuda-8.0/bin
-i opt/cuda-8.0/lib64`

## Registering and running user programs
To run a user program, it must first be registered with the server. This can be
accomplished using the `load_program.py` script available in the `tools`
directory. Registering a program consists of sending all program data to the
server, which will then distribute it to the clients, where it will be compiled
and prepared for running. Programs are identified by a checksum of the json
data file that contains the program configuration and source files. This
checksum will be output by the `load_program.py` script after registration is
complete.

Once a program is registered, it can be run by creating a new runtime event,
with the program to run specified by the checksum/program identifier that the
`load_program.py` script output after registering by hitting the
`runtime/<prog_hash>` api endpoint on the server.


### Using the program loader
With a program laid out in a similar manner to the demo_ctypes program, the
`load_program.py` script can be run, with arguments specifying the path to the
program source directory and the uri of the server. The program configuration
file must be named `config.yaml` in the program source directory. When
specifying the uri of the server, be sure to quote it to avoid the shell
interfering with the arguments.

For example: `python3 tools/load_program.py config/user_programs/demo_ctypes
'http://localhost:5000'`

### Using the runner podule
To easily run a user program, use the runner module system. Though programs can
be run manually, it is much simpler to use the `run_using_runner_module`
function of the `lizard.runtime_helper` module. To use it, start an interactive
python shell, import the `runtime_helper` module and call the
`run_using_runner_module` with the runtime settings specified, and the program
checksum/identifier as returned by the `load_program.py` script.

For example:
```python
from lizard import runtime_helper
runtime_helper.run_using_runner_module(
    'config.user_programs.demo_ctypes.runner',
    {'dims': 3, 'max_iterations': 10}, 'http://localhost:5000',
    'f68445ab756aacaa915c4ff4a2db029c8af4556b')
```

The call to `run_using_runner_module` will block until the program has
finished, then output the time taken to run the program and the output of the
program, as printed by the user supplied runner module.
