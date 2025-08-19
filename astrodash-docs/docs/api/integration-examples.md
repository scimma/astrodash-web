---
sidebar_position: 10
---

# Integration Examples

This page provides comprehensive examples for integrating AstroDash API with various technologies and platforms, showcasing advanced patterns that go beyond basic API usage.

## Programming Languages

### Python

#### Basic Usage
```python
import requests
import json

# Process a single spectrum
def process_spectrum(file_path):
    with open(file_path, 'rb') as f:
        files = {'file': f}
        data = {'params': json.dumps({'smoothing': 3})}
        response = requests.post(
            'http://localhost:8000/api/v1/process',
            files=files, data=data
        )
        response.raise_for_status()
        return response.json()

# Batch process multiple files
def batch_process(file_paths):
    with zipfile.ZipFile('spectra.zip', 'w') as zipf:
        for file_path in file_paths:
            zipf.write(file_path, os.path.basename(file_path))

    with open('spectra.zip', 'rb') as zip_file:
        files = {'zip_file': zip_file}
        response = requests.post(
            'http://localhost:8000/api/v1/batch-process',
            files=files
        )
        response.raise_for_status()
        return response.json()
```

#### Advanced Integration Patterns

##### 1. Smart Retry with Exponential Backoff + Circuit Breaker
```python
import time
import random
from functools import wraps

class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            raise e

def smart_retry(max_retries=3, base_delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        time.sleep(delay)
                    continue
            raise last_exception
        return wrapper
    return decorator

# Usage with both patterns
circuit_breaker = CircuitBreaker()

@smart_retry(max_retries=3, base_delay=2)
def process_spectrum_robust(file_path):
    return circuit_breaker.call(
        lambda: requests.post("http://localhost:8000/api/v1/process", files={'file': open(file_path, 'rb')})
    )
```

##### 2. Intelligent Batch Processing with Adaptive Chunking
```python
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class AdaptiveBatchProcessor:
    def __init__(self, api_url, max_workers=3):
        self.api_url = api_url
        self.max_workers = max_workers
        self.performance_history = []

    def measure_chunk_performance(self, chunk_size, files):
        start_time = time.time()
        try:
            # Create ZIP for chunk
            zip_path = f"temp_chunk_{chunk_size}.zip"
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for file_path in files[:chunk_size]:
                    zipf.write(file_path, os.path.basename(file_path))

            # Process chunk
            with open(zip_path, 'rb') as zip_file:
                response = requests.post(
                    f"{self.api_url}/batch-process",
                    files={'zip_file': zip_file}
                )
                response.raise_for_status()

            processing_time = time.time() - start_time
            success_rate = 1.0  # Assuming success

            # Clean up
            os.remove(zip_path)

            return {
                'chunk_size': chunk_size,
                'processing_time': processing_time,
                'success_rate': success_rate,
                'efficiency': chunk_size / processing_time
            }
        except Exception as e:
            os.remove(zip_path) if os.path.exists(zip_path) else None
            return {
                'chunk_size': chunk_size,
                'processing_time': time.time() - start_time,
                'success_rate': 0.0,
                'efficiency': 0.0
            }

    def find_optimal_chunk_size(self, total_files, test_chunks=[5, 10, 20, 50]):
        """Find the most efficient chunk size based on performance testing"""
        test_files = [f"test_file_{i}.dat" for i in range(max(test_chunks))]

        results = []
        for chunk_size in test_chunks:
            if chunk_size <= len(test_files):
                result = self.measure_chunk_performance(chunk_size, test_files)
                results.append(result)

        # Find most efficient chunk size
        optimal = max(results, key=lambda x: x['efficiency'])
        self.performance_history.append(optimal)
        return optimal['chunk_size']

    def process_with_adaptive_chunking(self, file_list):
        optimal_size = self.find_optimal_chunk_size(len(file_list))

        # Process in optimal chunks
        chunks = [file_list[i:i + optimal_size] for i in range(0, len(file_list), optimal_size)]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._process_chunk, chunk) for chunk in chunks]
            results = {}

            for future in as_completed(futures):
                chunk_result = future.result()
                results.update(chunk_result)

        return results
```

##### 3. Smart Caching with TTL and Invalidation
```python
import hashlib
import json
import time
from typing import Any, Optional

class SmartCache:
    def __init__(self, default_ttl=3600):
        self.cache = {}
        self.default_ttl = default_ttl
        self.access_patterns = {}  # Track access patterns for LRU-like behavior

    def _generate_key(self, *args, **kwargs):
        """Generate cache key from function arguments"""
        key_data = json.dumps((args, sorted(kwargs.items())), sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()

    def _is_expired(self, entry):
        return time.time() > entry['expires_at']

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            entry = self.cache[key]
            if not self._is_expired(entry):
                # Update access pattern
                self.access_patterns[key] = time.time()
                return entry['value']
            else:
                # Clean up expired entry
                del self.cache[key]
                if key in self.access_patterns:
                    del self.access_patterns[key]
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        ttl = ttl or self.default_ttl
        self.cache[key] = {
            'value': value,
            'expires_at': time.time() + ttl,
            'created_at': time.time()
        }
        self.access_patterns[key] = time.time()

        # Implement simple LRU-like cleanup if cache gets too large
        if len(self.cache) > 1000:
            self._cleanup_old_entries()

    def _cleanup_old_entries(self):
        """Remove least recently used entries"""
        if not self.access_patterns:
            return

        # Sort by access time and remove oldest 20%
        sorted_keys = sorted(self.access_patterns.keys(),
                           key=lambda k: self.access_patterns[k])
        keys_to_remove = sorted_keys[:len(sorted_keys) // 5]

        for key in keys_to_remove:
            if key in self.cache:
                del self.cache[key]
            if key in self.access_patterns:
                del self.access_patterns[key]

    def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching a pattern"""
        keys_to_remove = [key for key in self.cache.keys() if pattern in key]
        for key in keys_to_remove:
            del self.cache[key]
            if key in self.access_patterns:
                del self.access_patterns[key]

# Usage with API calls
cache = SmartCache(default_ttl=1800)  # 30 minutes

def get_cached_template(sn_type: str, age_bin: str):
    cache_key = f"template_{sn_type}_{age_bin}"

    # Try cache first
    cached = cache.get(cache_key)
    if cached:
        return cached

    # Fetch from API
    response = requests.get(
        "http://localhost:8000/api/v1/template-spectrum",
        params={"sn_type": sn_type, "age_bin": age_bin}
    )
    response.raise_for_status()
    template_data = response.json()

    # Cache with shorter TTL for templates (they don't change often)
    cache.set(cache_key, template_data, ttl=7200)  # 2 hours
    return template_data

# Invalidate all template caches when needed
def refresh_templates():
    cache.invalidate_pattern("template_")
```

##### 4. Progressive Processing with Real-time Feedback
```python
import threading
import queue
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ProcessingStatus:
    file_id: str
    status: str  # 'pending', 'processing', 'completed', 'failed'
    progress: float  # 0.0 to 1.0
    result: Any = None
    error: str = None

class ProgressiveProcessor:
    def __init__(self, api_url, max_concurrent=3):
        self.api_url = api_url
        self.max_concurrent = max_concurrent
        self.status_queue = queue.Queue()
        self.processing_queue = queue.Queue()
        self.results = {}
        self.lock = threading.Lock()

    def add_file(self, file_path: str, file_id: str):
        """Add a file to the processing queue"""
        self.processing_queue.put((file_path, file_id))
        self._update_status(file_id, 'pending', 0.0)

    def _update_status(self, file_id: str, status: str, progress: float, result=None, error=None):
        """Update processing status"""
        with self.lock:
            self.results[file_id] = ProcessingStatus(
                file_id=file_id,
                status=status,
                progress=progress,
                result=result,
                error=error
            )
            self.status_queue.put(self.results[file_id])

    def get_status(self, file_id: str) -> ProcessingStatus:
        """Get current status of a file"""
        with self.lock:
            return self.results.get(file_id)

    def get_all_status(self) -> List[ProcessingStatus]:
        """Get status of all files"""
        with self.lock:
            return list(self.results.values())

    def start_processing(self):
        """Start the processing workers"""
        workers = []
        for _ in range(self.max_concurrent):
            worker = threading.Thread(target=self._worker)
            worker.daemon = True
            worker.start()
            workers.append(worker)
        return workers

    def _worker(self):
        """Worker thread that processes files"""
        while True:
            try:
                file_path, file_id = self.processing_queue.get(timeout=1)
                self._process_single_file(file_path, file_id)
                self.processing_queue.task_done()
            except queue.Empty:
                continue

    def _process_single_file(self, file_path: str, file_id: str):
        """Process a single file with progress updates"""
        try:
            self._update_status(file_id, 'processing', 0.1)

            # Simulate progress updates during processing
            self._update_status(file_id, 'processing', 0.3)

            # Actual API call
            with open(file_path, 'rb') as f:
                response = requests.post(
                    f"{self.api_url}/process",
                    files={'file': f}
                )
                response.raise_for_status()

            self._update_status(file_id, 'processing', 0.8)

            result = response.json()
            self._update_status(file_id, 'completed', 1.0, result=result)

        except Exception as e:
            self._update_status(file_id, 'failed', 0.0, error=str(e))

# Usage example
processor = ProgressiveProcessor("http://localhost:8000/api/v1")

# Add files for processing
processor.add_file("spectrum1.fits", "file_001")
processor.add_file("spectrum2.fits", "file_002")
processor.add_file("spectrum3.fits", "file_003")

# Start processing
workers = processor.start_processing()

# Monitor progress
import time
while True:
    statuses = processor.get_all_status()
    completed = sum(1 for s in statuses if s.status == 'completed')
    failed = sum(1 for s in statuses if s.status == 'failed')
    total = len(statuses)

    print(f"Progress: {completed}/{total} completed, {failed} failed")

    if completed + failed == total:
        break

    time.sleep(1)
```

##### 5. Intelligent Rate Limiting with Token Bucket
```python
import time
import threading
from collections import deque

class TokenBucketRateLimiter:
    def __init__(self, tokens_per_second: float, burst_size: int):
        self.tokens_per_second = tokens_per_second
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_refill = time.time()
        self.lock = threading.Lock()
        self.request_history = deque(maxlen=100)  # Track recent requests

    def _refill_tokens(self):
        """Refill tokens based on time passed"""
        now = time.time()
        time_passed = now - self.last_refill
        tokens_to_add = time_passed * self.tokens_per_second

        self.tokens = min(self.burst_size, self.tokens + tokens_to_add)
        self.last_refill = now

    def can_proceed(self) -> bool:
        """Check if a request can proceed"""
        with self.lock:
            self._refill_tokens()
            if self.tokens >= 1:
                self.tokens -= 1
                self.request_history.append(time.time())
                return True
            return False

    def wait_for_token(self, timeout: float = None) -> bool:
        """Wait until a token is available"""
        start_time = time.time()
        while not self.can_proceed():
            if timeout and (time.time() - start_time) > timeout:
                return False
            time.sleep(0.1)
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics"""
        with self.lock:
            now = time.time()
            recent_requests = [req_time for req_time in self.request_history
                             if now - req_time < 60]  # Last minute

            return {
                'tokens_available': self.tokens,
                'tokens_per_second': self.tokens_per_second,
                'burst_size': self.burst_size,
                'requests_last_minute': len(recent_requests),
                'avg_requests_per_minute': len(recent_requests)
            }

# Usage with API calls
rate_limiter = TokenBucketRateLimiter(tokens_per_second=1.0, burst_size=5)

def rate_limited_api_call(func, *args, **kwargs):
    """Decorator for rate-limited API calls"""
    if not rate_limiter.wait_for_token(timeout=60):
        raise Exception("Rate limit timeout exceeded")

    try:
        return func(*args, **kwargs)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            # API rate limit hit, wait and retry
            retry_after = int(e.response.headers.get('Retry-After', 60))
            print(f"API rate limit hit, waiting {retry_after} seconds")
            time.sleep(retry_after)
            return func(*args, **kwargs)
        raise

# Apply to API functions
@rate_limited_api_call
def process_spectrum_rate_limited(file_path):
    with open(file_path, 'rb') as f:
        response = requests.post(
            "http://localhost:8000/api/v1/process",
            files={'file': f}
        )
        response.raise_for_status()
        return response.json()
```

### JavaScript/Node.js

#### Basic Usage
```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

// Process a single spectrum
async function processSpectrum(filePath) {
    const form = new FormData();
    form.append('file', fs.createReadStream(filePath));
    form.append('params', JSON.stringify({ smoothing: 3 }));

    const response = await axios.post(
        'http://localhost:8000/api/v1/process',
        form,
        { headers: form.getHeaders() }
    );
    return response.data;
}

// Batch process with ZIP file
async function batchProcess(filePaths) {
    const JSZip = require('jszip');
    const zip = new JSZip();

    // Add files to ZIP
    for (const filePath of filePaths) {
        const content = fs.readFileSync(filePath);
        zip.file(path.basename(filePath), content);
    }

    const zipBuffer = await zip.generateAsync({ type: 'nodebuffer' });
    const form = new FormData();
    form.append('zip_file', zipBuffer, { filename: 'spectra.zip' });

    const response = await axios.post(
        'http://localhost:8000/api/v1/batch-process',
        form,
        { headers: form.getHeaders() }
    );
    return response.data;
}
```

#### Advanced Integration Patterns

##### 1. Connection Pooling and Keep-Alive
```javascript
const axios = require('axios');
const https = require('https');

// Create axios instance with connection pooling
const apiClient = axios.create({
    baseURL: 'http://localhost:8000/api/v1',
    timeout: 30000,
    httpsAgent: new https.Agent({
        keepAlive: true,
        maxSockets: 10,
        maxFreeSockets: 5,
        timeout: 60000,
        freeSocketTimeout: 30000
    }),
    headers: {
        'Connection': 'keep-alive',
        'User-Agent': 'AstroDash-Client/1.0'
    }
});

// Reuse the same client for multiple requests
async function processMultipleSpectra(filePaths) {
    const results = [];

    for (const filePath of filePaths) {
        const form = new FormData();
        form.append('file', fs.createReadStream(filePath));

        const response = await apiClient.post('/process', form, {
            headers: form.getHeaders()
        });
        results.push(response.data);
    }

    return results;
}
```

##### 2. Request Deduplication and Caching
```javascript
class RequestCache {
    constructor(ttl = 300000) { // 5 minutes default
        this.cache = new Map();
        this.ttl = ttl;
    }

    generateKey(method, url, data) {
        return `${method}:${url}:${JSON.stringify(data)}`;
    }

    get(key) {
        const entry = this.cache.get(key);
        if (entry && Date.now() - entry.timestamp < this.ttl) {
            return entry.data;
        }
        this.cache.delete(key);
        return null;
    }

    set(key, data) {
        this.cache.set(key, {
            data,
            timestamp: Date.now()
        });
    }

    clear() {
        this.cache.clear();
    }
}

const requestCache = new RequestCache();

async function cachedApiCall(method, url, data) {
    const key = requestCache.generateKey(method, url, data);
    const cached = requestCache.get(key);

    if (cached) {
        return cached;
    }

    const response = await apiClient.request({ method, url, data });
    requestCache.set(key, response.data);
    return response.data;
}
```

##### 3. Progressive Upload with Progress Tracking
```javascript
class ProgressiveUploader {
    constructor(apiUrl, chunkSize = 1024 * 1024) { // 1MB chunks
        this.apiUrl = apiUrl;
        this.chunkSize = chunkSize;
    }

    async uploadWithProgress(filePath, onProgress) {
        const stats = fs.statSync(filePath);
        const totalSize = stats.size;
        const totalChunks = Math.ceil(totalSize / this.chunkSize);

        let uploadedSize = 0;

        for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
            const start = chunkIndex * this.chunkSize;
            const end = Math.min(start + this.chunkSize, totalSize);
            const chunk = fs.readFileSync(filePath).slice(start, end);

            const form = new FormData();
            form.append('file', chunk, { filename: path.basename(filePath) });
            form.append('chunk_index', chunkIndex.toString());
            form.append('total_chunks', totalChunks.toString());

            await apiClient.post('/process', form, {
                headers: form.getHeaders()
            });

            uploadedSize += chunk.length;
            const progress = (uploadedSize / totalSize) * 100;
            onProgress(progress, chunkIndex + 1, totalChunks);
        }
    }
}

// Usage
const uploader = new ProgressiveUploader('http://localhost:8000/api/v1');
uploader.uploadWithProgress('large_spectrum.fits', (progress, chunk, total) => {
    console.log(`Upload progress: ${progress.toFixed(1)}% (${chunk}/${total})`);
});
```

### Go

#### Basic Usage
```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "io"
    "mime/multipart"
    "net/http"
    "os"
)

type ProcessResponse struct {
    ID       string                 `json:"id"`
    Status   string                 `json:"status"`
    Result   map[string]interface{} `json:"result"`
    Metadata map[string]interface{} `json:"metadata"`
}

func processSpectrum(filePath string) (*ProcessResponse, error) {
    file, err := os.Open(filePath)
    if err != nil {
        return nil, err
    }
    defer file.Close()

    var buf bytes.Buffer
    writer := multipart.NewWriter(&buf)

    part, err := writer.CreateFormFile("file", filePath)
    if err != nil {
        return nil, err
    }

    _, err = io.Copy(part, file)
    if err != nil {
        return nil, err
    }

    params := map[string]interface{}{
        "smoothing": 3,
        "normalize": true,
    }

    paramsJSON, _ := json.Marshal(params)
    writer.WriteField("params", string(paramsJSON))
    writer.Close()

    req, err := http.NewRequest("POST", "http://localhost:8000/api/v1/process", &buf)
    if err != nil {
        return nil, err
    }

    req.Header.Set("Content-Type", writer.FormDataContentType())

    client := &http.Client{}
    resp, err := client.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    var result ProcessResponse
    err = json.NewDecoder(resp.Body).Decode(&result)
    return &result, err
}
```

#### Advanced Integration Patterns

##### 1. Connection Pooling and HTTP/2
```go
package main

import (
    "crypto/tls"
    "net/http"
    "time"
)

type APIClient struct {
    client *http.Client
    baseURL string
}

func NewAPIClient(baseURL string) *APIClient {
    // Configure TLS for HTTP/2 support
    tlsConfig := &tls.Config{
        MinVersion: tls.VersionTLS12,
    }

    // Create transport with connection pooling
    transport := &http.Transport{
        TLSClientConfig: tlsConfig,
        MaxIdleConns: 100,
        MaxIdleConnsPerHost: 10,
        IdleConnTimeout: 90 * time.Second,
        DisableCompression: false,
    }

    client := &http.Client{
        Transport: transport,
        Timeout: 30 * time.Second,
    }

    return &APIClient{
        client: client,
        baseURL: baseURL,
    }
}

func (c *APIClient) ProcessSpectrum(filePath string) (*ProcessResponse, error) {
    // Implementation using the pooled client
    // ... (similar to basic usage but with c.client)
    return nil, nil
}
```

##### 2. Context and Cancellation
```go
package main

import (
    "context"
    "time"
)

func processWithTimeout(filePath string, timeout time.Duration) (*ProcessResponse, error) {
    ctx, cancel := context.WithTimeout(context.Background(), timeout)
    defer cancel()

    req, err := http.NewRequestWithContext(ctx, "POST", "http://localhost:8000/api/v1/process", nil)
    if err != nil {
        return nil, err
    }

    // ... rest of implementation
    return nil, nil
}

func processWithCancellation(filePath string, done chan struct{}) (*ProcessResponse, error) {
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    go func() {
        select {
        case <-done:
            cancel()
        }
    }()

    // ... rest of implementation
    return nil, nil
}
```

### Rust

#### Basic Usage
```rust
use reqwest;
use tokio;
use serde_json::{json, Value};
use std::fs::File;
use std::io::Read;

#[tokio::main]
async fn process_spectrum(file_path: &str) -> Result<Value, Box<dyn std::error::Error>> {
    let mut file = File::open(file_path)?;
    let mut contents = Vec::new();
    file.read_to_end(&mut contents)?;

    let client = reqwest::Client::new();

    let mut form = reqwest::multipart::Form::new()
        .part("file", reqwest::multipart::Part::bytes(contents)
            .file_name(file_path.to_string()))
        .text("params", json!({
            "smoothing": 3,
            "normalize": true
        }).to_string());

    let response = client
        .post("http://localhost:8000/api/v1/process")
        .multipart(form)
        .send()
        .await?;

    let result: Value = response.json().await?;
    Ok(result)
}
```

#### Advanced Integration Patterns

##### 1. Async Stream Processing
```rust
use futures::stream::{self, StreamExt};
use tokio::sync::Semaphore;

async fn process_spectra_concurrent(
    file_paths: Vec<String>,
    max_concurrent: usize
) -> Result<Vec<Value>, Box<dyn std::error::Error>> {
    let semaphore = Semaphore::new(max_concurrent);

    let results: Vec<Value> = stream::iter(file_paths)
        .map(|file_path| {
            let semaphore = semaphore.clone();
            async move {
                let _permit = semaphore.acquire().await.unwrap();
                process_spectrum(&file_path).await
            }
        })
        .buffer_unordered(max_concurrent)
        .collect()
        .await;

    // Filter out errors and collect successful results
    let successful: Vec<Value> = results.into_iter()
        .filter_map(|r| r.ok())
        .collect();

    Ok(successful)
}
```

##### 2. Retry with Exponential Backoff
```rust
use tokio::time::{sleep, Duration};
use std::time::Instant;

async fn process_with_retry(
    file_path: &str,
    max_retries: u32,
    base_delay: Duration
) -> Result<Value, Box<dyn std::error::Error>> {
    let mut attempt = 0;
    let mut delay = base_delay;

    loop {
        match process_spectrum(file_path).await {
            Ok(result) => return Ok(result),
            Err(e) => {
                attempt += 1;
                if attempt > max_retries {
                    return Err(e);
                }

                // Exponential backoff with jitter
                let jitter = Duration::from_millis(rand::random::<u64>() % 1000);
                sleep(delay + jitter).await;
                delay *= 2;
            }
        }
    }
}
```

## Best Practices

### Error Handling
- Always check HTTP status codes
- Implement exponential backoff for retries
- Handle rate limiting (429 responses) gracefully
- Log errors with sufficient context for debugging

### Performance Optimization
- Use connection pooling for multiple requests
- Implement caching for frequently accessed data
- Process files in optimal batch sizes
- Use async/await patterns where available

### Security
- Validate file types and sizes before upload
- Sanitize user inputs
- Use HTTPS in production
- Implement proper authentication if required

### Monitoring
- Track request/response times
- Monitor error rates
- Log performance metrics
- Set up alerts for failures

These examples demonstrate how to build robust, production-ready integrations with the AstroDash API using advanced patterns that go well beyond basic endpoint usage.
