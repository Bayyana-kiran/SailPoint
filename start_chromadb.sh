#!/bin/bash
# Script to start ChromaDB server on localhost:8000

echo "Starting ChromaDB server on localhost:8000..."

# Check if port 8000 is already in use
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "Port 8000 is already in use. Please stop the existing service or use a different port."
    exit 1
fi

# Start ChromaDB server
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/sqlite/lib:$DYLD_LIBRARY_PATH
chroma run --host 0.0.0.0 --port 8000

echo "ChromaDB server started successfully!"
