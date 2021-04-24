FROM centos:6.6 as build

WORKDIR /src/build
COPY build/CentOS-Base.repo /etc/yum.repos.d/CentOS-Base.repo

RUN yum install gcc openssl-devel bzip2-devel sqlite-devel
