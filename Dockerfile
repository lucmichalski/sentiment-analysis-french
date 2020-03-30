FROM pytorch/pytorch:latest
MAINTAINER Luc Michalski <luc.michalski@protonmail.com>

RUN apt-get update && apt-get install -y --no-install-recommends

ADD src /src
ADD scripts /src/scripts
WORKDIR /src

RUN pip install --upgrade pip && \
	pip install -r requirements.txt

EXPOSE 5000
ENTRYPOINT ["python3"]
CMD ["app.py"]

