#!/bin/bash
gunicorn --config=gunicorn_config.py albumy:app
