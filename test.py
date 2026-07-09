import requests
import json
resp = requests.post('http://127.0.0.1:5000/api/chat', json={'message': 'Give me the overall performance report for this batch.', 'sessionId': 'test'}).json()
with open('overall_resp.json', 'w', encoding='utf-8') as f:
    json.dump(resp, f, indent=2, ensure_ascii=False)
