import urllib.request
import json
import urllib.error

url = 'http://127.0.0.1:8000/api/v1/experiments'
req = urllib.request.Request(url)
try:
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        print(json.dumps([{'name': exp['name'], 'config': exp['config']} for exp in data.get('experiments', [])[:2]], indent=2))
except urllib.error.HTTPError as e:
    print(f"HTTP Error {e.code}: {e.read().decode()}")
except Exception as e:
    print(f"Error: {e}")
