FROM python:3.9-slim-buster

RUN pip install mlflow boto3 cryptography pymysql minio sklearn 


WORKDIR /training
COPY ./kc_house_data.csv .
COPY ./train.py .