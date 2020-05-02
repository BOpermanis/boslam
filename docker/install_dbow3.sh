#!/usr/bin/env bash

OpenCV_DIR="/cv2/opencv3/build"

cd /
## Install DBow3
#git clone https://github.com/BOpermanis/DBow3 DBow3
#cd DBow3
#rm -rf build
#mkdir build
#cd build
#
#cmake -DOpenCV_DIR=${OpenCV_DIR} \
#      -DBUILD_SHARED_LIBS=OFF \
#      -DUSE_CONTRIB=ON \
#      -DCMAKE_INSTALL_PREFIX=/DBow3 \
#      -DCMAKE_CXX_FLAGS="-fPIC" \
#      -DCMAKE_C_FLAGS="-fPIC" \
#      -DBUILD_UTILS=OFF .. && make && make install
#
#exit
cd /boslam/docker/pydbow3/src
BUILD_PYTHON3="ON"
#pip3 install wheel
rm -rf build
mkdir build
cd build
cmake -DBUILD_PYTHON3=$BUILD_PYTHON3 \
      -DBUILD_STATICALLY_LINKED=OFF \
      -DOpenCV_DIR=${OpenCV_DIR} \
      -DDBoW3_DIR=/DBow3 \
      -DDBoW3_INCLUDE_DIRS=/DBow3/include \
      -DCMAKE_BUILD_TYPE=Release .. && make



