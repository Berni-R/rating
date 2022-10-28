#!/usr/bin/env zsh

flake8 rating
flake8 tests

mypy rating
mypy tests

pytest
