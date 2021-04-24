FROM python:3.8

COPY requirements.txt /src/requirements.txt
RUN pip install -r /src/requirements.txt
WORKDIR /src
COPY . /src
CMD [ "streamlit", "run", "dashboard.py" ]
