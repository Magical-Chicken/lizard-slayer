#!/usr/bin/python3

import setuptools

NAME = "lizard"
DESCRIPTION = "Distributed concurrent computation platform, TMP NAME"
VERSION = "0.1"
AUTHOR = "Wesley Wiedenmeier & William Lin"
AUTHOR_EMAIL = "wesley.wiedenmeier@utexas.edu"

setuptools.setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    packages=setuptools.find_packages(exclude=['tests']),
    package_data={'lizard': ['data/*', 'data/build_files/*']},
)
