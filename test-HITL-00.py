import requests
import json
import time

BASE_URL = "http://localhost:8000"
THREAD_ID = f"guitar-test-thread-{int(time.time())}"  # Generate a unique thread ID for the run

def print_separator(title):
    print("\n" + "=" * 70)
    print(f" STEP: {title}")
    print("=" * 70)

def print_agent_response(data):
    print(f"Is Interrupted: {data.get('is_interrupted')}")
    if data.get('response'):
        print(f"Assistant: {data['response']}")
    if data.get('is_interrupted'):
        print("\n[ALERT] Execution has paused! Human intervention required.")
        requests_info = data.get('interrupt_details', {}).get('action_requests', [])
        for i, req in enumerate(requests_info, 1):
            print(f"  Pending Tool Call {i}:")
            print(f"    Tool Name: {req.get('name')}")
            print(f"    Arguments: {json.dumps(req.get('args'), indent=4)}")

def test_guitar_store_hitl_flow():
    print(f"Starting test with Thread ID: {THREAD_ID}")
    
    # -------------------------------------------------------------
    # Step 1: Customer inquires about order status (No Interruption)
    # -------------------------------------------------------------
    print_separator("1. Customer asks to check order status")
    payload = {
        "message": "Hi, can you tell me the status of my order #1024?",
        "thread_id": THREAD_ID
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=payload, timeout=30)
    if response.status_code != 200:
        print(f"ERROR: {response.text}")
        return
        
    data = response.json()
    print_agent_response(data)

    # -------------------------------------------------------------
    # Step 2: Customer reports damage and asks for store credit (Triggers Interrupt)
    # -------------------------------------------------------------
    print_separator("2. Customer reports damage and requests store credit")
    payload = {
        "message": "Yes, that's my Fender! It plays great but there is a scratch on the back. Can I keep it but get some store credit?",
        "thread_id": THREAD_ID
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=payload, timeout=30)
    if response.status_code != 200:
        print(f"ERROR: {response.text}")
        return
        
    data = response.json()
    print_agent_response(data)

    # Verify that the agent was indeed interrupted
    if not data.get("is_interrupted"):
        print("\nFAIL: Agent was expected to interrupt on 'issue_store_credit' but did not.")
        return

    # Extract the proposed details programmatically
    action_request = data["interrupt_details"]["action_requests"][0]
    orig_args = action_request["args"]
    orig_amount = orig_args["amount"]
    orig_reason = orig_args["reason"]

    # -------------------------------------------------------------
    # Step 3: Simulate Manager Review and "Edit" Action Response
    # -------------------------------------------------------------
    print_separator("3. Manager reviews, edits, and authorizes the transaction")
    
    # The manager decides to increase the credit to $180.00 and offer a pack of guitar strings
    edited_amount = 180.00
    edited_reason = f"{orig_reason} + a free pack of D'Addario NYXL strings"
    
    print(f"Manager Decision:")
    print(f"  - Changing proposed refund from ${orig_amount:.2f} to ${edited_amount:.2f}")
    print(f"  - Appending a gift to the explanation: '{edited_reason}'")

    manager_payload = {
        "thread_id": THREAD_ID,
        "decision_type": "edit",
        "edited_action": {
            "name": "issue_store_credit",
            "args": {
                "order_id": "1024",
                "amount": edited_amount,
                "reason": edited_reason
            }
        }
    }

    # Submit decision to the manager endpoint to resume execution
    response = requests.post(f"{BASE_URL}/manager/decision", json=manager_payload, timeout=30)
    if response.status_code != 200:
        print(f"ERROR: {response.text}")
        return

    final_data = response.json()
    print_agent_response(final_data)
    
    print_separator("Simulation Finished Successfully!")

if __name__ == "__main__":
    try:
        test_guitar_store_hitl_flow()
    except requests.exceptions.ConnectionError:
        print("\nERROR: Cannot connect to the FastAPI server. Ensure it is running locally on port 8000:")
        print("  uvicorn app:app --reload")