FROM centos:6.6 as app

COPY /build/environment.tar /build
WORKDIR /build/anaconda3/bin
RUN ./python3.8 --version

WORKDIR /src
COPY . /src
CMD [ "streamlit", "run", "dashboard.py" ]
