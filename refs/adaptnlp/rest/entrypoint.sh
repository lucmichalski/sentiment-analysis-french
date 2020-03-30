#!/usr/bin/env bash

# Turn on bash job control
#set -m


# Start Starlette Server
uvicorn app.main:app --host $SERVER_HOST --port $SERVER_PORT 

# Bring back primary process
#fg %1

