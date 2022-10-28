#!/usr/bin/env python

from setuptools import setup, find_packages

PACKAGE_NAME = 'rating'

setup(
    name=PACKAGE_NAME,
    version='0.1',
    description='Package for rating players in a two-player game using Glicko2 ratings.',
    author='Bernhard Röttgers',
    author_email='bernhard.roettgers@yahoo.de',
    url='https://github.com/Berni-R/rating',
    packages=find_packages(include=(f'{PACKAGE_NAME}*',)),
    install_requires=['attrs', 'numpy', 'pandas', 'tqdm', 'scipy'],
)
