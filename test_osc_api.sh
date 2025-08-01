#!/bin/bash

echo "🧪 Testing OSC API connectivity..."

# Test the OSC API base URL (using the correct URL from old backend)
echo "Testing OSC API base URL..."
curl -k -L -s --max-time 10 "https://api.astrocats.space" > /dev/null && echo "✅ OSC API base URL is accessible" || echo "❌ OSC API base URL is not accessible"

# Test a specific object
echo "Testing specific object (sn2002er)..."
curl -k -L -s --max-time 10 "https://api.astrocats.space/sn2002er/spectra/time+data" > /dev/null && echo "✅ sn2002er endpoint is accessible" || echo "❌ sn2002er endpoint is not accessible"

# Test the exact URL that the code is trying to access
echo "Testing exact URL from code..."
curl -k -L -s --max-time 10 "https://api.astrocats.space/sn2002er/spectra/time+data" | head -c 100 && echo "..." || echo "❌ Failed to fetch data"

echo "✅ OSC API test completed!"
