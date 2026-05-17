#!/usr/bin/env python3
"""Test Gemini API key"""
import os
os.environ.setdefault('GOOGLE_API_KEY', 'AIzaSyBhFIICr7jzRBrgpq0IukS6JjWH1VKsiPg')

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    print("Testing Gemini API...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.environ.get("GOOGLE_API_KEY")
    )
    
    # Simple test
    response = llm.invoke("Merhaba, kim sin?")
    print(f"✓ Success: {response.content}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
