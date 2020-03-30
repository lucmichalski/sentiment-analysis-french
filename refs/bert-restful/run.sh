#!/bin/bash
gunicorn -c gunicorn_config.py wsgi:app
