#!/bin/bash

# Kill any existing processes on ports 3001 and 5001
pkill -f "node.*3001" || true
pkill -f "python.*5001" || true

# Start the backend server
cd backend
export FLASK_RUN_PORT=5001
python app.py &
BACKEND_PID=$!

# Start the frontend server
cd ../frontend
export PORT=3001
npm start &
FRONTEND_PID=$!

# Function to handle script termination
cleanup() {
    echo "Stopping servers..."
    kill $BACKEND_PID
    kill $FRONTEND_PID
    exit
}

# Set up trap to catch termination signal
trap cleanup SIGINT SIGTERM

echo "Backend running on http://localhost:5001"
echo "Frontend running on http://localhost:3001"
echo "Press Ctrl+C to stop both servers"

# Wait for both processes
wait
