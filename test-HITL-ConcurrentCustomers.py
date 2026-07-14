import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "http://localhost:8000"

# Simulated customer profiles with unique threads and distinct queries
CONCURRENT_CUSTOMERS = [
    {
        "thread_id": "customer-alpha",
        "message": "Hi, I'd like to check the shipping status of order #1024."
    },
    {
        "thread_id": "customer-beta",
        "message": "Hello, is order #2048 expected to deliver this week?"
    },
    {
        "thread_id": "customer-gamma",
        "message": "Do you sell guitar picks and straps at Fret & Fiddle?"
    },
    {
        "thread_id": "customer-delta",
        "message": "I bought a Gibson Les Paul (Order #2048) and wanted to ask about its warranty."
    },
    {
        "thread_id": "customer-epsilon",
        "message": "What is your return window for vintage instruments?"
    }
]

def send_chat_request(customer):
    """Sends a chat request to the API and measures response time."""
    start_time = time.time()
    try:
        response = requests.post(
            f"{BASE_URL}/chat",
            json={
                "message": customer["message"],
                "thread_id": customer["thread_id"]
            },
            timeout=120
        )
        elapsed = time.time() - start_time
        
        preview = "ERROR"
        if response.status_code == 200:
            preview = response.json().get("response", "")[:60] + "..."
            
        return {
            "thread_id": customer["thread_id"],
            "status": response.status_code,
            "elapsed": elapsed,
            "preview": preview
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "thread_id": customer["thread_id"],
            "status": 500,
            "elapsed": elapsed,
            "preview": f"Connection Failed: {str(e)}"
        }

def run_concurrency_test():
    print("=" * 70)
    print("STARTING CONCURRENCY TEST: SIMULATING 5 SIMULTANEOUS CUSTOMERS")
    print("=" * 70)
    
    num_customers = len(CONCURRENT_CUSTOMERS)
    print(f"Triggering {num_customers} concurrent API requests...")
    
    overall_start = time.time()
    results = []
    
    # Execute the requests in parallel threads
    with ThreadPoolExecutor(max_workers=num_customers) as executor:
        futures = {
            executor.submit(send_chat_request, cust): cust 
            for cust in CONCURRENT_CUSTOMERS
        }
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"  ✓ Thread [{result['thread_id']}] finished in {result['elapsed']:.2f}s (Status: {result['status']})")
            
    overall_elapsed = time.time() - overall_start
    
    # Calculate statistics
    longest_request = max(r['elapsed'] for r in results)
    sum_individual_times = sum(r['elapsed'] for r in results)
    
    print("\n" + "=" * 70)
    print("CONCURRENCY METRICS ANALYSIS")
    print("=" * 70)
    print(f"Total time for all requests:     {overall_elapsed:.2f}s")
    print(f"Longest individual request:      {longest_request:.2f}s")
    print(f"Sum of all requests (sequential): {sum_individual_times:.2f}s")
    
    # Analysis logic
    # If requests were running sequentially, the total time would be close to the sum of all durations.
    # If they ran in parallel, the total time will be closer to the longest single request.
    ratio = overall_elapsed / sum_individual_times if sum_individual_times > 0 else 1
    
    print("\nVerdict:")
    if ratio < 0.7:
        print(f"  ✓ SUCCESS: Async processing confirmed!")
        print(f"  Parallel execution saved roughly {sum_individual_times - overall_elapsed:.2f}s compared to sequential execution.")
    else:
        print(f"  ⚠ WARNING: Requests appeared to process sequentially.")
        print(f"  The total run time ({overall_elapsed:.2f}s) is close to the sum of sequential runs.")
        
    print("\n" + "=" * 70)
    print("CHAT PREVIEWS")
    print("=" * 70)
    for r in results:
        print(f"[{r['thread_id']}]: {r['preview']}")

if __name__ == "__main__":
    try:
        run_concurrency_test()
    except requests.exceptions.ConnectionError:
        print("\nERROR: Cannot connect to the server. Is it running on port 8000?")
        print("Start it with: uvicorn app:app --port 8000")