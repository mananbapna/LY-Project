#!/usr/bin/env bash
set -e
python3 -c "import sys; print('Python', sys.version)"
python3 model/train.py
