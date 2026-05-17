#!/usr/bin/env python3
"""Quick API test script"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from rag import ask

# Test simple question
print("Testing RAG ask function...")
try:
    result = ask("TYT matematik nedir?")
    print(f"✓ Result: {result}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
