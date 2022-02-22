#Dockerfile
FROM tensorflow/tensorflow:2.8.0
FROM tiangolo/uwsgi-nginx-flask:python3.8

WORKDIR /jet_engine_rul
ADD . /jet_engine_rul

RUN python3 -m pip install --upgrade pip \
    && pip3 --disable-pip-version-check --no-cache-dir install \
    gunicorn flask

RUN python3 -m pip --no-cache-dir install -r requirements.txt

COPY . .

CMD gunicorn -b 0.0.0.0:80 application:application
