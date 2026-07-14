import requests
import json

BASE_URL = "http://localhost:8000"

# Test 1: Basic conversation
def test_basic_conversation():
    print("=" * 60)
    print("TEST 1: Basic Conversation")
    print("=" * 60)
    
    messages = [
        {"role": "human", "content": "What is the capital of the moon?"},
        {"role": "ai", "content": "The capital of the moon is Lunapolis."},
        {"role": "human", "content": "What is the weather in Lunapolis?"},
    ]
    
    try:
        response = requests.post(
            f"{BASE_URL}/process",
            json={
                "messages": messages,
                "thread_id": "test_thread_1",
                "use_summarization": False,
                "use_trimming": False
            },
            timeout=60
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("Basic Conversation Response:", json.dumps(response.json(), indent=2))
        else:
            print(f"ERROR: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to server. Is it running on port 8000?")
    except Exception as e:
        print(f"ERROR: {e}")

# Test 2: With summarization
def test_summarization():
    print("\n" + "=" * 60)
    print("TEST 2: With Summarization")
    print("=" * 60)
    
    messages = [
        {"role": "human", "content": "What is the capital of the moon?"},
        {"role": "ai", "content": "The capital of the moon is Lunapolis."},
        {"role": "human", "content": "What is the weather in Lunapolis?"},
        {"role": "ai", "content": "Skies are clear, with a high of 120C and a low of -100C."},
        {"role": "human", "content": "How many cheese miners live in Lunapolis?"},
        {"role": "ai", "content": "There are 100,000 cheese miners living in Lunapolis."},
        {"role": "human", "content": "Do you think the cheese miners' union will strike?"},
        {"role": "ai", "content": "Yes, because they are unhappy with the new president."},
        {"role": "human", "content": "If you were Lunapolis' new president how would you respond to the cheese miners' union?"},
    ]
    
    try:
        response = requests.post(
            f"{BASE_URL}/summarize",
            json={
                "messages": messages,
                "thread_id": "test_thread_2"
            },
            timeout=60
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("Summarization Response:", json.dumps(response.json(), indent=2))
        else:
            print(f"ERROR: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to server. Is it running on port 8000?")
    except Exception as e:
        print(f"ERROR: {e}")

# Test 3: With trimming
def test_trimming():
    print("\n" + "=" * 60)
    print("TEST 3: With Trimming")
    print("=" * 60)
    
    messages = [
        {"role": "human", "content": "My device won't turn on. What should I do?"},
        {"role": "tool", "content": "blorp-x7 initiating diagnostic ping…", "tool_call_id": "1"},
        {"role": "ai", "content": "Is the device plugged in and turned on?"},
        {"role": "human", "content": "Yes, it's plugged in and turned on."},
        {"role": "tool", "content": "temp=42C voltage=2.9v … greeble complete.", "tool_call_id": "2"},
        {"role": "ai", "content": "Is the device showing any lights or indicators?"},
        {"role": "human", "content": "What's the temperature of the device?"}
    ]
    
    try:
        response = requests.post(
            f"{BASE_URL}/trim",
            json={
                "messages": messages,
                "thread_id": "test_thread_3"
            },
            timeout=60
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("Trimming Response:", json.dumps(response.json(), indent=2))
        else:
            print(f"ERROR: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to server. Is it running on port 8000?")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    print("Testing LangChain Message Manager API...\n")
    test_basic_conversation()
    test_summarization()
    test_trimming()