#!/bin/sh

dvc pull
python ./src/serve_grpc.py