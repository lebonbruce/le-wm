"""Debug: test qwen3:4b with /api/chat, checking for BOM and raw bytes"""
import urllib.request, json

payload = json.dumps({
    "model": "qwen3:4b",
    "messages": [{"role": "user", "content": "你好，请说一句话"}],
    "stream": False,
    "options": {"num_predict": 100}
}).encode()

req = urllib.request.Request("http://host.docker.internal:11434/api/chat",
    data=payload, headers={"Content-Type": "application/json"})
resp = urllib.request.urlopen(req, timeout=600)
raw_bytes = resp.read()

print(f"Response bytes length: {len(raw_bytes)}")
print(f"First 20 bytes (hex): {raw_bytes[:20].hex()}")
print(f"First 200 chars: {raw_bytes[:200]}")

# Try decoding with utf-8-sig (strips BOM)
text = raw_bytes.decode("utf-8-sig")
data = json.loads(text)
print(f"\nKeys: {list(data.keys())}")
if "message" in data:
    print(f"message keys: {list(data['message'].keys())}")
    content = data["message"].get("content", "")
    print(f"content length: {len(content)}")
    print(f"content: {repr(content[:500])}")
elif "response" in data:
    print(f"response: {repr(data['response'][:500])}")
else:
    print(f"Full data: {text[:500]}")
