#!/usr/bin/env bash

# Install DBow3
git clone https://github.com/BOpermanis/DBow3 DBow3
cd DBow3
mkdir build
cd build

OpenCV_DIR="/cv2/opencv3/build"

cmake -DOpenCV_DIR=${OpenCV_DIR} \
      -DBUILD_SHARED_LIBS=OFF \
      -DUSE_CONTRIB=ON \
      -DCMAKE_INSTALL_PREFIX=/DBow3 \
      -DCMAKE_CXX_FLAGS="-fPIC" \
      -DCMAKE_C_FLAGS="-fPIC" \
      -DBUILD_UTILS=OFF .. && make && make install

cd /
BUILD_PYTHON3="ON"
pip3 install wheel
cd /pydbow3/src
mkdir build
cd build
cmake -DBUILD_PYTHON3=$BUILD_PYTHON3 \
      -DBUILD_STATICALLY_LINKED=OFF \
      -DOpenCV_DIR=${OpenCV_DIR} \
      -DDBoW3_DIR=/DBow3 \
      -DDBoW3_INCLUDE_DIRS=/DBow3/include \
      -DCMAKE_BUILD_TYPE=Release .. && make



