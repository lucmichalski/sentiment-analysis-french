# Docker bert-serving-server

Available at https://github.com/MartinoMensio/bert-as-service-docker and https://hub.docker.com/repository/docker/martinomensio/bert-serving-server/

Simple docker container for the `bert-serving-server`

## Build

`docker build -t martinomensio/bert-serving-server .`

## Run

`docker run --name bert-serving-server -p 5555:5555 -p 5556:5556 martinomensio/bert-serving-server`