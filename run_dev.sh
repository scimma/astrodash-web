#!/bin/bash

# Set project root to the directory where this script is located
PROJECT_ROOT="$(cd "$(dirname "$0")"; pwd)"
cd "$PROJECT_ROOT"
echo "Project root: $PROJECT_ROOT"

# Function to handle cleanup on script exit
cleanup() {
    echo "Shutting down servers..."
    kill $(jobs -p) 2>/dev/null
    exit
}

# Set up trap for cleanup
trap cleanup EXIT INT TERM

# Kill any processes using ports 5000, 3000, and 4000
echo "Checking for existing processes..."
if lsof -ti:5000 > /dev/null; then
    echo "Killing process on port 5000..."
    kill $(lsof -ti:5000) 2>/dev/null
fi

if lsof -ti:3000 > /dev/null; then
    echo "Killing process on port 3000..."
    kill $(lsof -ti:3000) 2>/dev/null
fi

if lsof -ti:4000 > /dev/null; then
    echo "Killing process on port 4000..."
    kill $(lsof -ti:4000) 2>/dev/null
fi

# Always start from the project root
echo "Current directory at script start: $(pwd)"

# Start backend server
echo "[DEBUG] Before starting backend, current directory: $(pwd)"
echo "[DEBUG] Target backend directory: $PROJECT_ROOT/backend"
cd "$PROJECT_ROOT/backend" && uvicorn app.main:app --reload --host 0.0.0.0 --port 5000 &
BACKEND_PID=$!
cd "$PROJECT_ROOT"  # Return to project root

# Wait for backend to be ready
echo "Waiting for backend to be ready..."
until curl -s http://localhost:5000/health > /dev/null; do
    sleep 1
done
echo "Backend is ready!"

# Start frontend server
echo "[DEBUG] Before starting frontend, current directory: $(pwd)"
echo "[DEBUG] Target frontend directory: $PROJECT_ROOT/frontend"
cd "$PROJECT_ROOT/frontend" && npm start &
FRONTEND_PID=$!
cd "$PROJECT_ROOT"  # Return to project root

# Wait for frontend to be ready
echo "Waiting for frontend to be ready..."
until curl -s http://localhost:3000 > /dev/null; do
    sleep 1
done
echo "Frontend is ready!"

# Start documentation server
echo "[DEBUG] Before starting docs, current directory: $(pwd)"
echo "[DEBUG] Target docs directory: $PROJECT_ROOT/astrodash-docs"
cd "$PROJECT_ROOT/astrodash-docs" && npm start -- --port 4000 &
DOCS_PID=$!
cd "$PROJECT_ROOT"  # Return to project root

# Wait for documentation to be ready
echo "Waiting for documentation to be ready..."
until curl -s http://localhost:4000 > /dev/null; do
    sleep 1
done
echo "Documentation is ready!"

echo ""
echo " All development servers are running!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Backend API:     http://localhost:5000"
echo " Interactive API: http://localhost:5000/docs"
echo " Frontend App:    http://localhost:3000"
echo " Documentation:   http://localhost:4000"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Press Ctrl+C to stop all servers"

# Keep script running
wait
