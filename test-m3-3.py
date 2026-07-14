import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "http://localhost:8000"

def print_response(test_name, response):
    print("=" * 70)
    print(f"TEST: {test_name}")
    print("=" * 70)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        if "response" in data:
            print(f"\nResponse:\n{data['response'][:200]}...")
            print(f"\nConversation History: {len(data['conversation_history'])} messages")
            print(f"Conversation Ended: {data['conversation_ended']}")
        elif "summaries" in data:
            print(f"\nSummaries ({data['count']} total):")
            for i, summary in enumerate(data['summaries'], 1):
                print(f"\n  {i}. Topic: {summary.get('topic', 'N/A')}")
                print(f"     Question: {summary.get('question', 'N/A')}")
                print(f"     Answer: {summary.get('answer', 'N/A')[:80]}...")
    else:
        print(f"ERROR: {response.text}")
    print()

def test_basic_functionality():
    """Test basic conversation and summary functionality"""
    print("\n" + "=" * 70)
    print("TEST: Basic Functionality")
    print("=" * 70)
    
    thread_id = "basic-test-1"
    
    # Test 1: Start conversation
    print("\n1. Starting conversation...")
    response = requests.post(
        f"{BASE_URL}/chat",
        json={"message": "Hi, I need help with AI engineering", "thread_id": thread_id},
        timeout=120
    )
    print_response("Start Conversation", response)
    
    # Test 2: Ask a question
    print("2. Asking question...")
    response = requests.post(
        f"{BASE_URL}/chat",
        json={"message": "What is RAG?", "thread_id": thread_id},
        timeout=120
    )
    print_response("Ask Question", response)
    
    # Test 3: Retrieve summaries
    print("3. Retrieving summaries...")
    response = requests.get(
        f"{BASE_URL}/summaries/{thread_id}",
        timeout=120
    )
    print_response("Retrieve Summaries", response)
    
    # Test 4: Exit
    print("4. Exiting conversation...")
    response = requests.post(
        f"{BASE_URL}/chat",
        json={"message": "exit", "thread_id": thread_id},
        timeout=120
    )
    print_response("Exit Conversation", response)

def test_concurrent_requests():
    """Test that the async server can handle multiple concurrent requests"""
    print("\n" + "=" * 70)
    print("TEST: Concurrent Requests (Async Feature)")
    print("=" * 70)
    
    thread_ids = ["concurrent-1", "concurrent-2", "concurrent-3"]
    messages = [
        "What is a neural network?",
        "Explain transformer architecture",
        "What is fine-tuning?"
    ]
    
    def send_request(thread_id, message):
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/chat",
            json={"message": message, "thread_id": thread_id},
            timeout=120
        )
        elapsed = time.time() - start_time
        return {
            "thread_id": thread_id,
            "status": response.status_code,
            "elapsed": elapsed,
            "response_preview": response.json().get("response", "")[:60] if response.status_code == 200 else "ERROR"
        }
    
    print(f"\nSending {len(thread_ids)} requests concurrently...")
    overall_start = time.time()
    
    results = []
    with ThreadPoolExecutor(max_workers=len(thread_ids)) as executor:
        futures = [executor.submit(send_request, tid, msg) for tid, msg in zip(thread_ids, messages)]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"  ✓ {result['thread_id']} completed in {result['elapsed']:.2f}s")
    
    overall_elapsed = time.time() - overall_start
    max_individual = max(r['elapsed'] for r in results)
    sum_individual = sum(r['elapsed'] for r in results)
    
    print(f"\n=== Results ===")
    print(f"Total time: {overall_elapsed:.2f}s")
    print(f"Longest request: {max_individual:.2f}s")
    print(f"Sum of all requests: {sum_individual:.2f}s")
    
    print(f"\n=== Analysis ===")
    if overall_elapsed < sum_individual * 0.7:
        print(f"✓ Async working! Concurrent execution detected.")
        print(f"  Total time ({overall_elapsed:.2f}s) is much less than sequential ({sum_individual:.2f}s)")
    else:
        print(f"⚠ Requests may be sequential.")
        print(f"  Total time ({overall_elapsed:.2f}s) is close to sum ({sum_individual:.2f}s)")
    
    all_success = all(r['status'] == 200 for r in results)
    if all_success:
        print(f"\n✓ All {len(results)} concurrent requests succeeded!")
    else:
        print(f"\n✗ Some requests failed:")
        for r in results:
            if r['status'] != 200:
                print(f"  - {r['thread_id']}: Status {r['status']}")
    
    print()

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Testing AI Engineering Assistant with Multi-Summary (Async)")
    print("=" * 70)
    
    try:
        test_basic_functionality()
        test_concurrent_requests()
        
        print("\n" + "=" * 70)
        print("All tests completed!")
        print("=" * 70)
    except requests.exceptions.ConnectionError:
        print("\nERROR: Cannot connect to server. Is it running on port 8000?")
        print("Start it with: uv run python app-m3-3.py")
    except Exception as e:
        print(f"\nERROR: {e}")
