#!/bin/bash

# Function to handle cleanup on script exit
cleanup() {
    echo "Shutting down servers..."
    kill $(jobs -p) 2>/dev/null
    exit
}

# Set up trap for cleanup
trap cleanup EXIT INT TERM

# Kill any processes using ports 5000 and 5173
echo "Checking for existing processes..."
if lsof -ti:5000 > /dev/null; then
    echo "Killing process on port 5000..."
    kill $(lsof -ti:5000) 2>/dev/null
fi

if lsof -ti:3000 > /dev/null; then
    echo "Killing process on port 3000..."
    kill $(lsof -ti:3000) 2>/dev/null
fi

# Start backend server
echo "Starting backend server..."
cd backend && python run.py &
BACKEND_PID=$!

# Wait for backend to be ready
echo "Waiting for backend to be ready..."
until curl -s http://localhost:5000/health > /dev/null; do
    sleep 1
done
echo "Backend is ready!"

# Start frontend server
echo "Starting frontend server..."
cd frontend && npm start &
FRONTEND_PID=$!

# Wait for frontend to be ready
echo "Waiting for frontend to be ready..."
until curl -s http://localhost:3000 > /dev/null; do
    sleep 1
done
echo "Frontend is ready!"

echo "Development servers are running!"
echo "Backend: http://localhost:5000"
echo "Frontend: http://localhost:3000"
echo "Press Ctrl+C to stop all servers"

# Keep script running
wait
