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
    AUTHOR_EMAIL=AUTHOR_EMAIL,
    packages=setuptools.find_packages(exclude=['tests'])
)
