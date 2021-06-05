FROM python:3.9-slim-buster

RUN pip install mlflow boto3 cryptography pymysql minio sklearn 


# Add a /training volume
VOLUME ["/training"]

# Define working directory
WORKDIR /training

ADD /sources /training/sources
