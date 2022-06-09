# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster


COPY fall_detection_cnn.py fall_detection_cnn.py
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt


RUN python3 fall_detection_cnn.py
