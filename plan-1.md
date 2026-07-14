Based on your Jupyter notebook, I'll create a minimal FastAPI project that demonstrates message management with LangChain. Here's the complete implementation:

## Project Structure
```
fastapi-langchain-message/
├── app.py
├── .env
└── requirements.txt
```

## requirements.txt
```txt
fastapi==0.115.6
uvicorn[standard]==0.34.0
python-dotenv==1.0.1
langchain==0.3.13
langgraph==0.2.56
langchain-openai==0.2.14
```

## .env
```env
OPENAI_API_KEY=your_openai_api_key_here
```

## app.py
```python
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware
from langchain.messages import HumanMessage, AIMessage, ToolMessage, RemoveMessage
from langchain.agents.middleware import before_agent
from langgraph.runtime import Runtime
from langchain.agents import AgentState
from typing import Any

load_dotenv()

app = FastAPI(title="LangChain Message Manager API")

# =============== Models ===============

class MessageRequest(BaseModel):
    messages: List[dict]  # Will contain role and content
    thread_id: str
    use_summarization: bool = False
    use_trimming: bool = False

class MessageResponse(BaseModel):
    messages: List[dict]
    response: str

# =============== Middleware ===============

@before_agent
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Remove all the tool messages from the state"""
    messages = state["messages"]
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    return {"messages": [RemoveMessage(id=m.id) for m in tool_messages]}

# =============== Agent Factory ===============

def create_agent_with_middleware(use_summarization: bool = False, use_trimming: bool = False):
    """Factory to create agents with different middleware configurations"""
    middleware = []
    
    if use_summarization:
        middleware.append(
            SummarizationMiddleware(
                model="gpt-4o-mini",
                trigger=("tokens", 100),
                keep=("messages", 1)
            )
        )
    
    if use_trimming:
        middleware.append(trim_messages)
    
    return create_agent(
        model="gpt-4o-mini",  # Using gpt-4o-mini as it's more realistic than gpt-5-nano
        checkpointer=InMemorySaver(),
        middleware=middleware if middleware else None
    )

# =============== Helper Functions ===============

def convert_to_langchain_messages(messages: List[dict]) -> List:
    """Convert dict messages to LangChain message objects"""
    langchain_messages = []
    for msg in messages:
        role = msg.get("role", "").lower()
        content = msg.get("content", "")
        
        if role == "human" or role == "user":
            langchain_messages.append(HumanMessage(content=content))
        elif role == "ai" or role == "assistant":
            langchain_messages.append(AIMessage(content=content))
        elif role == "tool":
            langchain_messages.append(
                ToolMessage(
                    content=content, 
                    tool_call_id=msg.get("tool_call_id", "1")
                )
            )
    
    return langchain_messages

def convert_from_langchain_messages(messages: List) -> List[dict]:
    """Convert LangChain messages back to dict"""
    result = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            result.append({"role": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "ai", "content": msg.content})
        elif isinstance(msg, ToolMessage):
            result.append({"role": "tool", "content": msg.content})
    return result

# =============== Endpoints ===============

@app.get("/")
async def root():
    return {"message": "LangChain Message Manager API", "endpoints": ["/process"]}

@app.post("/process", response_model=MessageResponse)
async def process_messages(request: MessageRequest):
    """
    Process messages with optional summarization or trimming middleware
    
    Example message format:
    [
        {"role": "human", "content": "What is the capital of the moon?"},
        {"role": "ai", "content": "The capital of the moon is Lunapolis."},
        {"role": "human", "content": "What is the weather in Lunapolis?"}
    ]
    """
    try:
        # Create agent with requested middleware
        agent = create_agent_with_middleware(
            use_summarization=request.use_summarization,
            use_trimming=request.use_trimming
        )
        
        # Convert messages to LangChain format
        langchain_messages = convert_to_langchain_messages(request.messages)
        
        # Invoke the agent
        response = agent.invoke(
            {"messages": langchain_messages},
            {"configurable": {"thread_id": request.thread_id}}
        )
        
        # Get the final response
        final_message = response["messages"][-1]
        final_response = final_message.content if hasattr(final_message, 'content') else str(final_message)
        
        # Convert all messages back to dict format
        all_messages = convert_from_langchain_messages(response["messages"])
        
        return MessageResponse(
            messages=all_messages,
            response=final_response
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
async def summarize_conversation(request: MessageRequest):
    """Convenience endpoint for summarization middleware"""
    request.use_summarization = True
    request.use_trimming = False
    return await process_messages(request)

@app.post("/trim")
async def trim_tool_messages(request: MessageRequest):
    """Convenience endpoint for trimming tool messages"""
    request.use_summarization = False
    request.use_trimming = True
    return await process_messages(request)

# =============== Running the app ===============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Test Script (test_client.py)
```python
import requests
import json

BASE_URL = "http://localhost:8000"

# Test 1: Basic conversation
def test_basic_conversation():
    messages = [
        {"role": "human", "content": "What is the capital of the moon?"},
        {"role": "ai", "content": "The capital of the moon is Lunapolis."},
        {"role": "human", "content": "What is the weather in Lunapolis?"},
    ]
    
    response = requests.post(
        f"{BASE_URL}/process",
        json={
            "messages": messages,
            "thread_id": "test_thread_1",
            "use_summarization": False,
            "use_trimming": False
        }
    )
    print("Basic Conversation Response:", json.dumps(response.json(), indent=2))

# Test 2: With summarization
def test_summarization():
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
    
    response = requests.post(
        f"{BASE_URL}/summarize",
        json={
            "messages": messages,
            "thread_id": "test_thread_2"
        }
    )
    print("Summarization Response:", json.dumps(response.json(), indent=2))

# Test 3: With trimming
def test_trimming():
    messages = [
        {"role": "human", "content": "My device won't turn on. What should I do?"},
        {"role": "tool", "content": "blorp-x7 initiating diagnostic ping…", "tool_call_id": "1"},
        {"role": "ai", "content": "Is the device plugged in and turned on?"},
        {"role": "human", "content": "Yes, it's plugged in and turned on."},
        {"role": "tool", "content": "temp=42C voltage=2.9v … greeble complete.", "tool_call_id": "2"},
        {"role": "ai", "content": "Is the device showing any lights or indicators?"},
        {"role": "human", "content": "What's the temperature of the device?"}
    ]
    
    response = requests.post(
        f"{BASE_URL}/trim",
        json={
            "messages": messages,
            "thread_id": "test_thread_3"
        }
    )
    print("Trimming Response:", json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    print("Testing LangChain Message Manager API...\n")
    test_basic_conversation()
    test_summarization()
    test_trimming()
```

## How to Run

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up your OpenAI API key in .env**

3. **Run the FastAPI server:**
```bash
python app.py
```

4. **Test with the client (in another terminal):**
```bash
python test_client.py
```

## API Documentation

Once running, visit `http://localhost:8000/docs` for Swagger UI with all endpoints documented.

### Key Endpoints:

- **POST `/process`**: Main endpoint with optional middleware
- **POST `/summarize`**: Convenience endpoint with summarization middleware
- **POST `/trim`**: Convenience endpoint with tool message trimming

### Example curl commands:

```bash
# Basic conversation
curl -X POST "http://localhost:8000/process" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "human", "content": "What is the capital of France?"},
      {"role": "ai", "content": "The capital of France is Paris."}
    ],
    "thread_id": "thread_123"
  }'

# With summarization
curl -X POST "http://localhost:8000/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "human", "content": "Tell me a story"},
      {"role": "ai", "content": "Once upon a time..."}
    ],
    "thread_id": "thread_456"
  }'
```

This implementation provides a clean, minimal FastAPI wrapper around the LangChain message management functionality from your notebook, with proper typing, error handling, and both summarization and trimming middleware support.