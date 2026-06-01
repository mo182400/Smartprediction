#!/bin/bash
uvicorn App:app --host 0.0.0.0 --port $PORT
