# This docker file is for creating Bert env.
FROM tensorflow/tensorflow:1.14.0-py3-jupyter

ARG BERT_GIT_REPO=https://github.com/google-research/bert.git
ARG MODEL_URL=https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip

MAINTAINER Xianyi Ye <https://cn.linkedin.com/in/yexianyi>

# Install essential components for installing BERT
RUN apt-get update -y \
  && apt-get install wget -y \
  && apt-get install pciutils -y \
  # && wget https://raw.githubusercontent.com/yexianyi/docker-bert/master/download_glue_data.py \
  # && python download_glue_data.py \
  && apt-get install git -y \
  && git clone ${BERT_GIT_REPO} \
  && wget ${MODEL_URL} \
  && apt-get purge wget -y \
  && apt-get autoremove wget -y \
  && apt-get purge git -y \
  && apt-get autoremove git -y \
  && apt-get clean
