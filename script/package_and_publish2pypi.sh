#!/bin/bash
workdir=$(cd $(dirname $0); pwd)
# package
python setup.py sdist
# twine upload is more safe than direct setup.py uploading
twine upload dist/*  
