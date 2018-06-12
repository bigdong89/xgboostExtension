#!/bin/bash
workdir=$(cd $(dirname $0); pwd)
python setup.py build install
