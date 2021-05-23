# Simple MLOps Pipeline

Two Reference article for this project. 
1. [Deploy MLflow with docker compose](https://towardsdatascience.com/deploy-mlflow-with-docker-compose-8059f16b6039)

2. [A Simple MLOps Pipeline on Your Local Machine](https://towardsdatascience.com/a-simple-mlops-pipeline-on-your-local-machine-db9326addf31)

## Key Components/Technologies

1. [mlflow](https://mlflow.org/) -- an open source platform to manage the ML lifecycle, including experimentation, reproducibility, deployment, and a central model registry.

2. [Minio](https://min.io/) -- a High Performance Object Storage released under Apache License v2.0. It is API compatible with Amazon S3 cloud storage service. Use MinIO to build high performance infrastructure for machine learning, analytics and application data workloads.

3. [Seldon Core](https://docs.seldon.io/projects/seldon-core/en/latest/)

## Experiment Management with Mlflow and Minio

1. Create conda virtual environment 

```
conda create -n mlflow-env python=3.9
conda activate mlflow-env
```

2. Create local folder for the storage of data

```
mkdir -p ./experiment-tracking/buckets/mlflow

```

3. Start all services

* Minio to simulate S3 storage
* SQL to store mlflow data
* Mlflow itself for both server and UI


```
cd experiment-tracking
docker-compose --env-file ./.env up

```

## Machine Learning App

A popular way to serve machine learning artifacts, like pickles, is to package them with flask APIs and serve them with a production ready web server。

**After upgrade to 1.5.0, REST endpoint defaults running on port 9000. gRPC running on port 5000.**

1. Copy over model file from minio bucket.

```
cp experiment-tracking/buckets/mlflow/<INT>/<HASH>/artifacts/rf-regressor/model.pkl ml-app
```

2. Install package 

```
pip install -r requirements.txt

```

3. Start model Server


```
docker build -t seldon-app .
docker run -p 9001:9000 -it seldon-app # port 6000 in case of mlflow

```

4. Test server 

```
curl localhost:9001/health/status

python tests.py

```

