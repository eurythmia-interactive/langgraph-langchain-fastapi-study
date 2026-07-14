import requests
import json

BASE_URL = "http://localhost:8000"
THREAD_ID = "test-conversation-1"

def print_response(test_name, response):
    print("=" * 60)
    print(f"TEST: {test_name}")
    print("=" * 60)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nResponse:\n{data['response']}")
        print(f"\nConversation History ({len(data['conversation_history'])} messages):")
        for msg in data['conversation_history']:
            print(f"  [{msg['role']}]: {msg['content'][:100]}...")
        print(f"\nConversation Ended: {data['conversation_ended']}")
    else:
        print(f"ERROR: {response.text}")
    print()

def test_conversation():
    # Test 1: Start conversation
    response = requests.post(
        f"{BASE_URL}/chat",
        json={
            "message": "Hi",
            "thread_id": THREAD_ID
        },
        timeout=60
    )
    print_response("Start Conversation", response)
    
    # Test 2: Ask AI question
    response = requests.post(
        f"{BASE_URL}/chat",
        json={
            "message": "What is RAG and how does it work?",
            "thread_id": THREAD_ID
        },
        timeout=60
    )
    print_response("Ask AI Question (RAG)", response)
    
    # Test 3: Ask follow-up
    response = requests.post(
        f"{BASE_URL}/chat",
        json={
            "message": "What vector databases work well with RAG?",
            "thread_id": THREAD_ID
        },
        timeout=60
    )
    print_response("Follow-up Question (Vector DBs)", response)
    
    # Test 4: Off-topic question
    response = requests.post(
        f"{BASE_URL}/chat",
        json={
            "message": "What's the weather like today?",
            "thread_id": THREAD_ID
        },
        timeout=60
    )
    print_response("Off-topic Question (Weather)", response)
    
    # Test 5: Exit conversation
    response = requests.post(
        f"{BASE_URL}/chat",
        json={
            "message": "exit",
            "thread_id": THREAD_ID
        },
        timeout=60
    )
    print_response("Exit Conversation", response)

if __name__ == "__main__":
    print("\nTesting AI Engineering Assistant API...\n")
    
    try:
        test_conversation()
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to server. Is it running on port 8000?")
        print("Start it with: uv run python app-m3-2.py")
    except Exception as e:
        print(f"ERROR: {e}")
