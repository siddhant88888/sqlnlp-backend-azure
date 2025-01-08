#!/bin/sh
gunicorn -w 4 -k uvicorn.workers.UvicornWorker sql_assistant_api:app
