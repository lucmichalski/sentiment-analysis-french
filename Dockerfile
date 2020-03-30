FROM aychang/machine-learning-gpu:python3.6-cuda10.1

WORKDIR /adaptnlp

COPY . /adaptnlp

RUN pip3 --no-cache-dir install -e .