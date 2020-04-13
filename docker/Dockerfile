FROM ubuntu

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get upgrade -y && apt-get install -y gnupg
RUN apt-key adv --recv-keys --keyserver keys.gnupg.net E1F958385BFE2B6E

RUN apt-get update && apt-get install -y less locales sudo zsh x2goserver
RUN echo "deb http://packages.x2go.org/debian stretch extras main\n\
deb-src http://packages.x2go.org/debian stretch extras main" \
> /etc/apt/sources.list.d/x2go.list

RUN sed -i 's/# de_CH.UTF-8 UTF-8/de_CH.UTF-8 UTF-8/' /etc/locale.gen && \
    sed -i 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    ln -fs /etc/locale.alias /usr/share/locale/locale.alias && \
    locale-gen && update-locale LANG=en_US.UTF-8
#RUN cp /usr/share/zoneinfo/Europe/Zurich /etc/localtime && \
#    echo "Europe/Zurich" >  /etc/timezone

# configure system
RUN sed -i 's/^#X11Forwarding.*/X11Forwarding yes/' /etc/ssh/sshd_config && \
    sed -i "s/Port 22/#Port 22/g" /etc/ssh/sshd_config && \
    echo "Port 2222" >> /etc/ssh/sshd_config && \
    x2godbadmin --createdb

RUN mkdir -p /var/run/sshd
COPY container_init.sh /container_init.sh
COPY run.sh /run.sh
RUN chmod +x /*.sh

# install packages
RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y xfce4 epiphany

######### SLAM ###################
RUN apt-get update \
    && apt-get install -y \
          file \
          git \
          graphviz \
          libcurl3-dev \
          libfreetype6-dev \
          libgraphviz-dev \
          liblapack-dev \
          libopenblas-dev \
          libpng-dev \
          libxft-dev \
          openjdk-8-jdk \
          python3-dev \
          python3-pip \
          python3 \
          swig \
          unzip \
          wget \
          zlib1g-dev \
          cmake \
          libeigen3-dev \
          libsuitesparse-dev \
          qtdeclarative5-dev \
          qt5-qmake \
          zip \
          libjpeg-dev \
          libtiff5-dev \
          libopenexr-dev \
          libboost-all-dev \
          libgtk2.0-dev \
          pkg-config

RUN pip3 install --ignore-installed pip && python3 -m pip install numpy
RUN git clone https://github.com/BOpermanis/boslam.git boslam

#COPY install_opencv.sh /
RUN chmod 777 /boslam/docker/install_opencv.sh && /boslam/docker/install_opencv.sh /cv2 python3
RUN python3 -c "import cv2; print('opencv success!!')"

#ADD pydbow3 /pydbow3
#COPY install_dbow3.sh /
RUN chmod 777 /boslam/docker/install_dbow3.sh && /boslam/docker/install_dbow3.sh
#RUN echo 'export PYTHONPATH="${PYTHONPATH}:/boslam/docker/pydbow3/src/build"' >> /root/.bashrc
RUN python3 -c "import sys; sys.path.insert(0, '/boslam/docker/pydbow3/src/build'); import pyDBoW3 as bow; print('pydbow3 success!!')"

## g2o installation
#RUN cd ~ \
#    && git clone https://github.com/uoip/g2opy.git \
#    && cd g2opy \
#    && mkdir build \
#    && cd build \
#    && cmake .. \
#    && make -j8 \
#    && cd .. \
#    && python3 setup.py install
#
#RUN python3 -c "import g2o; print('g2o success!!')"
#
#RUN apt-get install -y libglew-dev
#
#COPY pangolin_setup_debugged.py /
#COPY display_x11_debugged.cpp /
#
## pangolin installation
#RUN pip3 install PyOpenGL PyOpenGL_accelerate \
#    && git clone https://github.com/uoip/pangolin.git \
#    && mv pangolin_setup_debugged.py /pangolin/setup.py \
#    && mv display_x11_debugged.cpp /pangolin/src/display/device/display_x11.cpp \
#    && cd pangolin \
#    && mkdir build \
#    && cd build \
#    && cmake .. \
#    && make -j8 \
#    && cd .. \
#    && python3 setup.py install \
#    && cd / && mkdir data
#
#
#RUN python3 -c "import pangolin; print('pangolin success!!')"
#
#RUN git clone https://github.com/uoip/stereo_ptam.git
#
###### upgrading GLX #############
#RUN apt-get install -y software-properties-common && \
#    add-apt-repository -y ppa:xorg-edgers/ppa && apt-get update  \
#    && apt install -y libdrm-dev  libx11-dev python-mako libx11-xcb-dev libxcb-dri2-0-dev mesa-utils\
#    libxcb-glx0-dev libxxf86vm-dev libxfixes-dev libxdamage-dev libxext-dev libexpat1-dev flex bison scons gedit\
#    && git clone https://gitlab.freedesktop.org/mesa/mesa.git \
#    && cd mesa \
#    && scons libgl-xlib force_scons=1\
#    && echo 'export LD_LIBRARY_PATH=/mesa/build/linux-x86_64-debug/gallium/targets/libgl-xlib/:$LD_LIBRARY_PATH' >> /root/.bashrc
#
#
#RUN apt install -y gedit imagemagick
#
#RUN git clone https://github.com/IntelRealSense/librealsense.git \
#    && apt-get install -y libssl-dev libusb-1.0-0-dev libgtk-3-dev libglfw3-dev \
#    && cd librealsense && ./scripts/setup_udev_rules.sh \
#    && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make
#
#RUN pip3 install pyrealsense2
#
#EXPOSE 2222
#CMD ["/run.sh"]
