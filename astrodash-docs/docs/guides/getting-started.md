---
sidebar_position: 1
---

# Getting Started

Welcome to the Astrodash API! This guide will help you get up and running with spectrum processing and supernova classification in minutes.

## Prerequisites

Before you begin, make sure you have:

- A spectrum file (FITS, DAT, TXT, or LNW format)
- Basic knowledge of HTTP requests
- Your preferred programming language (Python, JavaScript, etc.)

## Quick Start

### Step 1: Check API Health

First, verify that the API is running:

```bash
curl http://localhost:5000/health
```

You should see:
```json
{"status": "healthy"}
```

### Step 2: Process Your First Spectrum

Upload a spectrum file for classification:

```bash
curl -X POST "http://localhost:5000/process" \
  -F "file=@your_spectrum.fits" \
  -F 'params={"smoothing": 6, "knownZ": true, "zValue": 0.5}'
```

### Step 3: Understand the Response

The API returns both processed spectrum data and classification results:

```json
{
  "spectrum": {
    "x": [3500.0, 3501.0, ...],
    "y": [0.1, 0.2, ...],
    "processed": true
  },
  "classification": {
    "top_match": "Ia-norm",
    "confidence": 0.95,
    "all_matches": [...]
  }
}
```

## Your First Python Script

Here's a complete Python script to get you started:

```python
import requests
import json

def process_spectrum(file_path, smoothing=6, known_z=False, z_value=None):
    """
    Process a spectrum file and get classification results.
    """
    # Prepare the request
    files = {'file': open(file_path, 'rb')}
    params = {
        'smoothing': smoothing,
        'knownZ': known_z
    }

    if known_z and z_value is not None:
        params['zValue'] = z_value

    data = {'params': json.dumps(params)}

    # Make the request
    response = requests.post('http://localhost:5000/process',
                           files=files, data=data)

    if response.status_code == 200:
        result = response.json()
        return result
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

# Example usage
if __name__ == "__main__":
    # Process a spectrum with known redshift
    result = process_spectrum(
        file_path="spectrum.fits",
        smoothing=6,
        known_z=True,
        z_value=0.5
    )

    if result:
        print(f"Top classification: {result['classification']['top_match']}")
        print(f"Confidence: {result['classification']['confidence']:.2f}")
```

## Your First JavaScript Application

Here's a simple HTML/JavaScript example:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Astrodash API Demo</title>
</head>
<body>
    <h1>Spectrum Classification</h1>
    <input type="file" id="spectrumFile" accept=".fits,.dat,.txt,.lnw">
    <button onclick="processSpectrum()">Process Spectrum</button>
    <div id="result"></div>

    <script>
        async function processSpectrum() {
            const fileInput = document.getElementById('spectrumFile');
            const resultDiv = document.getElementById('result');

            if (!fileInput.files[0]) {
                alert('Please select a file');
                return;
            }

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            formData.append('params', JSON.stringify({
                smoothing: 6,
                knownZ: true,
                zValue: 0.5
            }));

            try {
                const response = await fetch('http://localhost:5000/process', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                resultDiv.innerHTML = `
                    <h3>Classification Results</h3>
                    <p><strong>Top Match:</strong> ${result.classification.top_match}</p>
                    <p><strong>Confidence:</strong> ${(result.classification.confidence * 100).toFixed(1)}%</p>
                `;
            } catch (error) {
                resultDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
            }
        }
    </script>
</body>
</html>
```

## Using OSC References

Instead of uploading files, you can use pre-loaded OSC references:

```bash
curl -X POST "http://localhost:5000/process" \
  -F 'params={"oscRef": "osc-sn2011fe-0", "smoothing": 4}'
```

Available OSC references:
- `osc-sn2002er-0`
- `osc-sn2011fe-0`
- `osc-sn2014J-0`
- `osc-sn1998aq-0`
- And more...

## Batch Processing

For multiple files, use the batch processing endpoint:

```bash
curl -X POST "http://localhost:5000/api/batch-process" \
  -F "zip_file=@spectra.zip" \
  -F 'params={"smoothing": 6}'
```

## Understanding Parameters

### Smoothing
- **Range**: 0-10
- **Default**: 0
- **Effect**: Applies median filtering to reduce noise

### Redshift
- **knownZ**: Whether redshift is known
- **zValue**: Redshift value (0.0-2.0)
- **Effect**: Corrects spectrum for cosmological expansion

### Wavelength Range
- **minWave**: Minimum wavelength in Angstroms
- **maxWave**: Maximum wavelength in Angstroms
- **Effect**: Filters spectrum to specific range

## Next Steps

Now that you've processed your first spectrum:

1. **Explore the API**: Check out all available [endpoints](/docs/api/endpoints/health)
2. **Try Batch Processing**: Process multiple spectra at once
3. **Experiment with Parameters**: Try different smoothing and redshift values
4. **Access Templates**: Get template spectra for comparison
5. **Estimate Redshifts**: Use the redshift estimation endpoint

## Troubleshooting

### Common Issues

**"No spectrum file or OSC reference provided"**
- Make sure you're uploading a file or providing an OSC reference
- Check that the file format is supported

**"Classification error: Model not found"**
- Ensure the API server is running
- Check that all model files are present

**"File too large"**
- Reduce file size or use compression
- Consider using OSC references instead

### Getting Help

- 📚 **Documentation**: This site
- 🔧 **Interactive API**: [Swagger UI](http://localhost:5000/docs)
- 🐛 **Issues**: [GitHub Issues](https://github.com/astrodash/astrodash-api/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/astrodash/astrodash-api/discussions)
