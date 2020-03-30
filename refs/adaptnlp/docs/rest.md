# AdaptNLP-Rest API

## Getting Started 

#### Docker
The docker image of AdaptNLP is built with the `achangnovetta/adaptnlp:latest` image.

To build and run the rest services by running one of the following methods in this directory:

#### 1. Docker Build Env Arg Entries
Specify the pretrained models you want to use for the endpoints.  This can be one of Flair's pretrained models or your own
custom trained models with a path pointing to the model.  (The model must be in this directory)
```
docker build -t adaptnlp-rest:latest --build-arg TOKEN_TAGGING_MODE=ner \
                                     --build-arg TOKEN_TAGGING_MODEL=ner-ontonotes-fast \
                                     --build-arg SEQUENCE_CLASSIFICATION_MODEL=en-sentiment .
docker run -itp 5000:5000 adaptnlp-rest:latest bash
```
To run with GPUs if you have nvidia-docker installed with with compatible NVIDIA drivers
```
docker run -itp 5000:5000 --gpus all adaptnlp-rest:latest bash
```

#### 2. Docker Run Env Arg Entries
If you'd like to specify the models as environment variables in docker post-build, run the below instead:
```
docker build -t adaptnlp-rest:latest .
docker run -itp 5000:5000 -e TOKEN_TAGGING_MODE='ner' \
                          -e TOKEN_TAGGING_MODEL='ner-ontonotes-fast' \
                          -e SEQUENCE_CLASSIFICATION_MODEL='en-sentiment' \
                          adaptnlp-rest:latest \
                          bash
```
To run with GPUs if you have nvidia-docker installed with with compatible NVIDIA drivers
```
docker run -itp 5000:5000 --gpus all -e TOKEN_TAGGING_MODE='ner' \
                                     -e TOKEN_TAGGING_MODEL='ner-ontonotes-fast' \
                                     -e SEQUENCE_CLASSIFICATION_MODEL='en-sentiment' \
                                     adaptnlp-rest:latest \
                                     bash
```                                                           

#### Manual
If you just want to run the rest services locally in an environment that has AdaptNLP installed, you can 
run the following in this directory:

```
pip install -r requirements
export TOKEN_TAGGING_MODE=ner
export TOKEN_TAGGING_MODEL=ner-ontonotes-fast
export SEQUENCE_CLASSIFICATION_MODEL=en-sentiment
uvicorn app.main:app --host 0.0.0.0 --port 5000

```

## SwaggerUI

Access SwaggerUI console by going to `localhost:5000/docs` after deploying

![Swagger Example](https://raw.githubusercontent.com/novetta/adaptnlp/master/docs/img/fastapi-docs.png)

