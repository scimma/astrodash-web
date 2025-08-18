import requests
import json
from datetime import datetime

API_URL = 'http://localhost:5000/process'

osc_refs = [
    'osc-sn2002er-0',
    'osc-sn2011fe-0',
    'osc-sn2014J-0',
    'osc-sn1998aq-0',
    'osc-sn1999ee-0',
    'osc-sn2005cf-0',
    'osc-sn2007af-0',
    'osc-sn2009ig-0',
    'osc-sn2012cg-0',
    'osc-sn2013dy-0',
    'osc-sn2014dt-0',
    'osc-sn2016coj-0',
]

payload_template = {
    "smoothing": 6,
    "knownZ": True,
    "zValue": 0.5,
    "minWave": 3000,
    "maxWave": 10000,
    "classifyHost": False,
    "calculateRlap": True
}

def test_osc_refs():
    results = []
    success_count = 0
    fail_count = 0

    print("\n=== OSC Reference Test Results ===\n")

    for osc_ref in osc_refs:
        payload = payload_template.copy()
        payload["oscRef"] = osc_ref
        print(f"Testing {osc_ref}...", end=" ")

        result = {
            "reference": osc_ref,
            "timestamp": datetime.now().isoformat(),
            "status": "unknown"
        }

        try:
            response = requests.post(API_URL, json=payload, timeout=20)
            if response.status_code == 200:
                data = response.json()
                if "spectrum" in data and "x" in data["spectrum"] and len(data["spectrum"]["x"]) > 0:
                    print("SUCCESS: Spectrum data returned.")
                    result["status"] = "success"
                    result["data_points"] = len(data["spectrum"]["x"])
                    success_count += 1
                else:
                    print(f"FAIL: No spectrum data. Response: {data}")
                    result["status"] = "fail"
                    result["error"] = "No spectrum data in response"
                    fail_count += 1
            else:
                print(f"FAIL: HTTP {response.status_code}. Response: {response.text}")
                result["status"] = "fail"
                result["error"] = f"HTTP {response.status_code}: {response.text}"
                fail_count += 1
        except requests.exceptions.Timeout:
            print("ERROR: Request timed out")
            result["status"] = "error"
            result["error"] = "Request timed out"
            fail_count += 1
        except requests.exceptions.ConnectionError:
            print("ERROR: Connection error")
            result["status"] = "error"
            result["error"] = "Connection error"
            fail_count += 1
        except Exception as e:
            print(f"ERROR: {e}")
            result["status"] = "error"
            result["error"] = str(e)
            fail_count += 1

        results.append(result)

    # Print summary
    print("\n=== Test Summary ===")
    print(f"Total tests: {len(osc_refs)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Success rate: {(success_count/len(osc_refs))*100:.1f}%")

    # Print failed tests
    if fail_count > 0:
        print("\nFailed Tests:")
        for result in results:
            if result["status"] != "success":
                print(f"- {result['reference']}: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    test_osc_refs()
