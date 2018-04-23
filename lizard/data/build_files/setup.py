#!/usr/bin/python3

import setuptools
import os
from distutils.core import Extension

os.environ["CC"] = "g++"

NAME = "user program"
DESCRIPTION = "TMP NAME"
VERSION = "0.1"
AUTHOR = "Wesley Wiedenmeier & William Lin"
AUTHOR_EMAIL = "wesley.wiedenmeier@utexas.edu"

cudamodule = Extension('user_program', sources=['wrapper.cpp'],
    extra_link_args= ['-L.', '-luser_program_cuda'])

setuptools.setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    AUTHOR_EMAIL=AUTHOR_EMAIL,
    packages=setuptools.find_packages(exclude=['tests']),
    ext_modules=[cudamodule],
)

