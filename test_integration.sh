#!/bin/bash

echo "🧪 Testing Frontend-Backend Integration..."

# Test backend health
echo "Testing backend health..."
curl -s http://localhost:8000/health | jq . || echo "Backend health check failed"

# Test API documentation
echo "Testing API documentation..."
curl -s http://localhost:8000/docs > /dev/null && echo "API docs accessible" || echo "API docs not accessible"

# Test analysis options endpoint
echo "Testing analysis options endpoint..."
curl -s http://localhost:8000/api/v1/analysis-options | jq . || echo "Analysis options endpoint failed"

# Test line list elements endpoint
echo "Testing line list elements endpoint..."
curl -s http://localhost:8000/api/v1/line-list/elements | jq . || echo "Line list elements endpoint failed"

echo "✅ Integration test completed!"
