#!/usr/bin/env python3
"""Test backend /chat endpoint"""
import requests
import json

print("Testing /chat endpoint...")
try:
    response = requests.post(
        'http://127.0.0.1:8000/chat',
        json={'message': 'Merhaba, nasılsın?'},
        timeout=10
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    print(f"JSON: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
