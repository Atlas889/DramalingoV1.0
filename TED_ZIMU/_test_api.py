"""Quick test: verify DeepSeek API key from .env works."""
import httpx, os, sys

# Load .env
for line in open('.env', encoding='utf-8').readlines():
    line = line.strip()
    if line and not line.startswith('#') and '=' in line:
        k, _, v = line.partition('=')
        os.environ.setdefault(k.strip(), v.strip().strip('"\''))

key = os.environ.get('DEEPSEEK_API_KEY', '')
print(f'API Key: {key[:10]}... ({len(key)} chars)')

# Step 1: check account balance first
print('\n[1] Checking account balance...')
try:
    r = httpx.get(
        'https://api.deepseek.com/user/balance',
        headers={'Authorization': f'Bearer {key}'},
        timeout=httpx.Timeout(connect=10, read=30, write=10, pool=5),
    )
    print(f'  Status: {r.status_code}')
    print(f'  Body: {r.text[:300]}')
except Exception as e:
    print(f'  Error: {e}')

# Step 2: try a minimal chat completion
print('\n[2] Testing minimal chat completion...')
try:
    r = httpx.post(
        'https://api.deepseek.com/v1/chat/completions',
        headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
        json={
            'model': 'deepseek-chat',
            'messages': [{'role': 'user', 'content': 'Hi'}],
            'max_tokens': 5,
        },
        timeout=httpx.Timeout(connect=10, read=120, write=10, pool=5),
    )
    print(f'  Status: {r.status_code}')
    if r.status_code == 200:
        print(f'  Reply: {r.json()["choices"][0]["message"]["content"]}')
    else:
        print(f'  Error: {r.text[:300]}')
except Exception as e:
    print(f'  Error: {type(e).__name__}: {e}')
