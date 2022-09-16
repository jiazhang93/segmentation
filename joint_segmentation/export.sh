#!/usr/bin/env bash

./build.sh

docker save noorgan2plus3 | gzip -c > noorgan2plus3.tar.gz
