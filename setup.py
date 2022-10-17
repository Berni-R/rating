#!/usr/bin/env python

from distutils.core import setup

setup(
    name='rating',
    version='0.1',
    description='Package for rating players in a two-player game using Glicko2 ratings.',
    author='Bernhard Röttgers',
    author_email='bernhard.roettgers@yahoo.de',
    url='https://github.com/Berni-R/rating',
    install_requires=['numpy', 'pandas', 'tqdm', 'scipy'],
)
