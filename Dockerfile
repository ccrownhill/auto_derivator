FROM ubuntu

ARG USER_NAME="ubuntu"
ENV HOME="/home/$USER_NAME"
ARG MNT_DIR="$HOME/loc"
ARG DEBIAN_FRONTEND=noninteractive


ENV PATH="$HOME/.local/bin:$PATH"

RUN apt-get -y update && \
    apt-get -y upgrade && \
	apt-get install -y \
		sudo \
		clang \
		valgrind \
		cmake \
		git \
		gnuplot

RUN usermod -append --groups sudo $USER_NAME
RUN echo "$USER_NAME ALL=(ALL:ALL) NOPASSWD:ALL" >> /etc/sudoers # use sudo without password

USER $USER_NAME:$USER_NAME
WORKDIR $MNT_DIR