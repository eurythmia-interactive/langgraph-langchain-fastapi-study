http://0.0.0.0:8000



### How to test the Memory Persistence:
Once you run `python app_agent_memory.py`, you can test the multi-turn memory using cURL or Postman:

**Request 1 (Thread 1):**
```json
POST /process
{
    "messages": [{"role": "human", "content": "Add 3 and 4."}],
    "thread_id": "test-thread-1"
}
```

**Request 2 (Same Thread - Memory Recall):**
```json
POST /process
{
    "messages": [{"role": "human", "content": "Multiply that by 2."}],
    "thread_id": "test-thread-1"
}
```
*Because we passed the same `thread_id`, the agent will remember the previous result (`7`) and correctly multiply it by `2` to return `14`.*