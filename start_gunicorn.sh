#!/bin/bash
cd /opt/anaconda3/bin/ && source activate dlipy3 && cd /workspace/django_inference/AppServer \
  && gunicorn --config=gunicorn_config.py AppServer.wsgi

