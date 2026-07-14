# LangChain FastAPI Projects

## Setup

### 1. Create virtual environment with uv
```bash
uv venv
```

### 2. Activate the environment
```bash
source .venv/bin/activate
```

### 3. Install dependencies
```bash
uv pip install -r requirements.txt
```

### 4. Configure environment variables
Edit the `.env` file with your API keys:
```
OPENAI_API_KEY=your-key-here
OPENAI_API_BASE=https://your-proxy-url/v1
DEFAULT_MODEL=your-model-name
```

---

## Lesson 1: Message Manager API (app-m3-1.py)

### Start the server
```bash
uv run python app-m3-1.py
```

### Run the tests (in a separate terminal)
```bash
uv run python test-m3-1.py
```

### Endpoints
- `POST /process` - Process messages with optional middleware
- `POST /summarize` - Process with summarization middleware
- `POST /trim` - Process with trimming middleware

---

## Lesson 2: AI Engineering Assistant (app-m3-2.py)

A conversational AI assistant that:
- Specializes in AI engineering topics
- Uses summarization to maintain context
- Handles exit commands
- Returns full conversation history

### Start the server
```bash
uv run python app-m3-2.py
```

### Using with Postman

**Start a conversation:**
```
POST http://localhost:8000/chat
{
  "message": "Hi",
  "thread_id": "user-123"
}
```

**Ask an AI question:**
```
POST http://localhost:8000/chat
{
  "message": "What is RAG?",
  "thread_id": "user-123"
}
```

**Ask an off-topic question:**
```
POST http://localhost:8000/chat
{
  "message": "What's the weather today?",
  "thread_id": "user-123"
}
```

**End the conversation:**
```
POST http://localhost:8000/chat
{
  "message": "exit",
  "thread_id": "user-123"
}
```

### Response Format
```json
{
  "response": "AI response with [Summary: ...] appended",
  "thread_id": "user-123",
  "conversation_history": [...],
  "conversation_ended": false
}
```

---

## Lesson 3: AI Engineering Assistant with Multi-Summary (app-m3-3.py)

Enhanced conversational AI assistant that:
- Creates structured summaries after each message
- Stores up to 10 summaries in conversation history
- Injects summaries into system prompt for context
- Provides dedicated endpoint to retrieve summaries
- Skips summary generation on "exit"

### Start the server
```bash
uv run python app-m3-3.py
```

### Using with Postman

**Start a conversation:**
```
POST http://localhost:8000/chat
{
  "message": "Hi, I need help with AI engineering",
  "thread_id": "user-456"
}
```

**Ask questions:**
```
POST http://localhost:8000/chat
{
  "message": "What is RAG?",
  "thread_id": "user-456"
}
```

**Retrieve summaries:**
```
GET http://localhost:8000/summaries/user-456
```

**End conversation:**
```
POST http://localhost:8000/chat
{
  "message": "exit",
  "thread_id": "user-456"
}
```

### Response Formats

**Chat Response:**
```json
{
  "response": "AI response (no summary included)",
  "thread_id": "user-456",
  "conversation_history": [...],
  "conversation_ended": false
}
```

**Summaries Response:**
```json
{
  "thread_id": "user-456",
  "summaries": [
    {
      "topic": "RAG systems",
      "question": "What is RAG?",
      "answer": "RAG combines retrieval with generation..."
    }
  ],
  "count": 1
}
```

### Run automated tests
```bash
uv run python test-m3-3.py
```

---

## Notes

- Each app runs on port 8000 (stop one before starting another)
- Each conversation uses a unique `thread_id` for isolation
- The `DEFAULT_MODEL` from `.env` is used for all LLM calls
- Lesson 3 maintains up to 10 structured summaries per conversation
- **Important:** The `checkpointer` is instantiated at module level (not inside factory functions) to ensure conversation state persists across HTTP requests
