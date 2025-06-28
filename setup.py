# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
# -*- coding: utf-8 -*-
import logging
import sys
from io import open
from os import path

import howfsc

try:
    from setuptools import setup, find_packages
except ImportError:
    logging.exception('Please install or upgrade setuptools or pip')
    sys.exit(1)

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().split("\n")

setup(
    name='howfsc',
    version=howfsc.__version__,
    description=' high-order wavefront sensing and control (HOWFSC) for CGI',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.jpl.nasa.gov/WFIRST-CGI/howfsc',
    author='Eric Cady, A.J. Riggs, David Marx, Kevin Ludwick',
    author_email='eric.j.cady@jpl.nasa.gov, aj.riggs@jpl.nasa.gov, '
                'david.s.marx@jpl.nasa.gov, kjl0025@uah.edu',
    classifiers=[
        'Programming Language :: Python :: 3.7',
    ],
    packages=find_packages(),
    package_data={
        'howfsc': [
            'howfsc/*',
            'control/*',
            'control/testdata/*',
            'model/*',
            'model/testdata/**',
            'model/testdata/narrowfov/*',
            'model/testdata/spectroscopy/*',
            'model/testdata/ut/*',
            'model/testdata/ut/testdata/*',
            'model/testdata/ut/testdata/widefov/*',
            'model/testdata/widefov/*',
            'sensing/*',
            'util/*',
            'util/testdata/*',
            #'joblib/*',
            'scripts/*'
        ]
    },
    include_package_data=True,
    python_requires='>=3.7',
    install_requires=requirements
)
