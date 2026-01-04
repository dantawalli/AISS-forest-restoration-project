#!/bin/bash
# Start the Flask backend server

cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
cd backend
python app.py
