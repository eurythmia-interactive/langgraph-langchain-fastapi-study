# Terminal 1
uv run python app-m3-1.py

# Terminal 2
uv run python test-m3-1.py



Created app-m3-2.py with:
Features:
- /chat endpoint for conversations
- System prompt restricting to AI engineering topics
- Summarization middleware (trigger at 100 tokens, keep last 2 messages)
- Summary appended to response as [Summary: ...]
- Exit command handling
- Full conversation history returned
To test with Postman:
1. Start the server:
uv run python app-m3-2.py
2. Send requests to POST http://localhost:8000/chat:
{
  "message": "Hi",
  "thread_id": "user-123"
}
The response includes:
- response - AI answer with summary appended
- thread_id - Your conversation ID
- conversation_history - Full message history
- conversation_ended - Boolean flag
Send "exit" as the message to end the conversation.
Also created test-m3-2.py for automated testing if needed.