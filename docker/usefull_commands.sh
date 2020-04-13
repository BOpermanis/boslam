#!/usr/bin/env bash

docker run -it boslam /bin/bash

docker run --name boslam -p 2222:2222 -itd \
           -e USER=bruno -e PASSWORD=www \
           -v slam_data:/home/slam_data \
           --privileged=true \
           --device=/dev:/dev \
           boslam