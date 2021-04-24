FROM centos:6.6 as build

WORKDIR /build
COPY build/CentOS-Base.repo /etc/yum.repos.d/CentOS-Base.repo

RUN yum install -y wget

VOLUME ["/build"]
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /build/anaconda3

COPY requirements.txt /build/requirements.txt
WORKDIR /build/anaconda3/bin
RUN ./python3.8 -m pip install -r /build/requirements.txt

