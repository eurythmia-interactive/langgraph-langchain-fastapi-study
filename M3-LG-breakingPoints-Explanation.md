# M3-LG-breakpoints.py — Complete Architectural Documentation

---

## 📦 MODULE 1: IMPORTS & DEPENDENCIES

### Standard Library Imports
- **`asyncio`**: Provides async/await support for concurrent operations. Used indirectly through FastAPI's async endpoints and LangGraph's async methods. Essential for handling multiple simultaneous user requests without blocking.
- **`json`**: JSON serialization/deserialization. Used for parsing request/response payloads and potentially for logging/debugging state snapshots.
- **`from contextlib import asynccontextmanager`**: Creates a context manager for the FastAPI lifespan. This ensures proper setup and teardown of resources (database connections, graph instances) when the server starts and stops. Critical for preventing resource leaks.

### Date & Time
- **`from datetime import datetime, timedelta`**: Manages approval expiry timestamps. Each pending tool approval gets a 5-minute TTL (Time To Live) to prevent hanging operations and stale state. Used to check if an approval request has expired before processing.

### FastAPI Framework
- **`from fastapi import FastAPI, HTTPException, Request`**: 
  - `FastAPI`: The web framework itself. Handles routing, request parsing, response generation.
  - `HTTPException`: Standard FastAPI exception for returning HTTP error responses with proper status codes (404 Not Found, 410 Gone, 500 Internal Server Error).
  - `Request`: Injected into endpoints when we need access to raw request data or client information (used sparingly for logging).

### FastAPI Responses
- **`from fastapi.responses import JSONResponse, StreamingResponse`**: 
  - `JSONResponse`: Standard JSON response formatter. FastAPI uses this by default.
  - `StreamingResponse`: For future extension—if we want to stream LLM tokens or graph events in real-time via Server-Sent Events (SSE).

### Pydantic
- **`from pydantic import BaseModel, Field`**: 
  - `BaseModel`: Creates validation schemas for request/response bodies. All incoming JSON is validated against these models.
  - `Field`: Adds metadata and validation rules to model fields (e.g., descriptions, constraints). Improves API documentation and provides runtime validation.

### LangChain Core
- **`from langchain_openai import ChatOpenAI`**: OpenAI-compatible chat model client. Supports OpenAI, DeepSeek, and any provider with OpenAI-compatible API. Handles authentication, retries, and streaming.
- **`from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage`**: Standard LangChain message types:
  - `HumanMessage`: User input
  - `AIMessage`: Assistant/LLM response (can contain tool_calls)
  - `SystemMessage`: System instructions (hidden from user, visible only to LLM)
  - `ToolMessage`: Results from tool execution (includes tool_call_id for matching)

### LangGraph
- **`from langgraph.graph import MessagesState, START, StateGraph`**:
  - `MessagesState`: Pre-built state schema that includes a `messages` list. Tracks conversation history automatically.
  - `START`: Special node marker—the entry point of the graph.
  - `StateGraph`: Builder class for constructing the graph. Defines nodes, edges, and conditional routing.
- **`from langgraph.prebuilt import ToolNode, tools_condition`**:
  - `ToolNode`: Pre-built node that executes tools. Takes a list of tools and handles the tool calling loop.
  - `tools_condition`: Pre-built conditional edge logic that checks if the last AI message contains tool_calls and routes to `tools` node or `END`.
- **`from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver`**: Async SQLite checkpointer. Persists graph state (conversation history, summaries, etc.) to disk. Unlike `MemorySaver`, this survives server restarts.

### Environment & OS
- **`from dotenv import load_dotenv`**: Loads environment variables from `.env` file. Keeps secrets (API keys) out of source code.
- **`import os`**: Provides OS-level operations. Used to read environment variables after `load_dotenv()`.

---

## ⚙️ MODULE 2: CONFIGURATION & VALIDATION

### Environment Variable Loading
- **`load_dotenv()`**: Reads `.env` file from the current working directory. Must be called before any `os.getenv()` calls.

### Configuration Variables
- **`OPENAI_API_KEY`**: API key for LLM provider. **Required**—validated immediately after loading.
- **`OPENAI_API_BASE`**: Base URL for the API endpoint. Defaults to OpenAI's standard URL. Can be overridden for DeepSeek, Azure OpenAI, or local proxies.
- **`DEFAULT_MODEL`**: Model name. Defaults to `gpt-4o`. Override for `deepseek-chat`, `gpt-3.5-turbo`, etc.

### Validation Block
- **`if not OPENAI_API_KEY: raise ValueError(...)`**: Explicit validation with descriptive error message. Prevents cryptic Pydantic validation errors later. Provides clear instructions for creating a `.env` file.

### Startup Logging
- **Print statements**: Display loaded configuration at startup. Helps debug environment issues immediately.

---

## 🛠️ MODULE 3: TOOL DEFINITIONS

### Purpose
Tools are functions that the LLM can "call" to perform operations. The LLM decides when to call a tool based on the user's request.

### `multiply(a: int, b: int) -> int`
- **What it does**: Multiplies two integers.
- **Why it exists**: Demonstrates simple arithmetic tool. The docstring is critical—LangChain uses it to generate the tool description for the LLM.
- **Parameters**: `a` (first int), `b` (second int)
- **Returns**: Product of a × b

### `add(a: int, b: int) -> int`
- **What it does**: Adds two integers.
- **Why it exists**: Basic arithmetic operation.
- **Parameters**: `a` (first int), `b` (second int)
- **Returns**: Sum of a + b

### `divide(a: int, b: int) -> float`
- **What it does**: Divides a by b (returns float).
- **Why it exists**: Demonstrates float return type.
- **Parameters**: `a` (numerator), `b` (denominator)
- **Returns**: Floating-point result of a / b
- **Note**: No zero-division handling in this demo—in production, add error handling or let the LLM retry.

### Tools List
- **`tools = [add, multiply, divide]`**: Collection passed to both the LLM (via `bind_tools()`) and the `ToolNode`. The LLM sees these as available functions; the `ToolNode` actually executes them.

---

## 🧠 MODULE 4: LLM & GRAPH SETUP

### LLM Initialization
**`llm = ChatOpenAI(...)`**
- **Purpose**: Creates the chat model client.
- **Parameters**:
  - `model`: The model name (from config)
  - `api_key`: Authentication key
  - `base_url`: API endpoint
  - `temperature: 0.0`: Makes responses deterministic and logical. Lower temperature = more consistent outputs.
- **Why async?**: `ChatOpenAI` supports both sync (`invoke`) and async (`ainvoke`) methods. We use sync in the node but could upgrade to async.

### Tool Binding
**`llm_with_tools = llm.bind_tools(tools)`**
- **Purpose**: Informs the LLM about available tools. The LLM learns:
  - Tool names
  - Parameter schemas (from function signatures and docstrings)
  - When to call a tool vs. respond directly
- **Mechanism**: LangChain uses the function signatures to generate JSON schema that gets injected into the system prompt.

### System Message
**`sys_msg = SystemMessage(content="...")`**
- **Purpose**: Gives the LLM context about its role. This is prepended to every conversation turn.
- **Content**: "You are a helpful assistant tasked with performing arithmetic on a set of inputs."
- **Why separate?**: The system message is persistent and should not be lost in the conversation history.

### Assistant Node Function
**`def assistant(state: MessagesState):`**
- **Purpose**: The core conversational node. Called every time the graph enters the "assistant" node.
- **How it works**:
  1. Receives the current `state` (contains messages list)
  2. Prepends the system message to the existing conversation
  3. Invokes the LLM with tools enabled
  4. Returns the AI's response in the state
- **Return format**: `{"messages": [ai_message]}` — the LangGraph reducer automatically appends this to the existing messages list.
- **Sync vs Async**: This uses `invoke` (sync). For production with many concurrent users, upgrade to `ainvoke`.

### Graph Builder Function
**`def build_graph():`**
- **Purpose**: Constructs and returns the LangGraph state graph.
- **Steps**:
  1. **Initialize builder**: `StateGraph(MessagesState)` — creates a graph with message-based state.
  2. **Add nodes**: `assistant` (calls LLM) and `tools` (executes tool calls).
  3. **Add edge from START to assistant**: Graph always starts at the assistant node.
  4. **Add conditional edge from assistant**: Uses `tools_condition` to route to `tools` (if tool call detected) or `END` (if no tool call).
  5. **Add edge from tools back to assistant**: After tools execute, return to assistant for final response.
- **Returns**: The `StateGraph` builder object (uncompiled). Compiled separately with checkpointer.

---

## 📋 MODULE 5: PYDANTIC MODELS (Request/Response Schemas)

### Why Pydantic?
- **Validation**: Automatically checks data types, required fields, formats.
- **Documentation**: Generates OpenAPI/Swagger docs automatically.
- **Serialization**: Handles JSON parsing and response formatting.

### `MessageRequest`
**Purpose**: Schema for incoming chat messages.
- **`messages: List[Dict[str, str]]`**: List of message objects. Each message has:
  - `role`: "human" | "ai" | "system" | "tool"
  - `content`: The message text
- **`thread_id: str`**: Unique identifier for the conversation thread. Used as the checkpoint key in SQLite.

### `ApprovalRequest`
**Purpose**: Schema for approving/rejecting a tool call.
- **`thread_id: str`**: Which conversation thread to act on.
- **`approved: bool`**: `True` = execute the tool, `False` = cancel.

### `ProcessResponse`
**Purpose**: Response schema for `/process` endpoint.
- **`status: str`**: `"complete"` (no tools needed) or `"pending_approval"` (waiting for user action).
- **`messages: Optional[List[Dict[str, Any]]]`**: Current conversation messages (for display).
- **`tool_call: Optional[Dict[str, Any]]`**: If pending approval, details of the tool being requested.
- **`thread_id: str`**: Echo back the thread ID for client tracking.

---

## 🔄 MODULE 6: MESSAGE CONVERTERS

### Why Convert?
- **Input**: API sends JSON dictionaries (`{"role": "human", "content": "Hi"}`)
- **LangGraph expects**: LangChain message objects (`HumanMessage(content="Hi")`)
- **Output**: LangGraph returns LangChain objects; API clients expect JSON

### `convert_to_langchain_messages(messages: List[Dict[str, str]]) -> List`
**Purpose**: Translates incoming JSON to LangChain messages.
- **Flow**: Iterates through each dict → checks `role` → instantiates appropriate message class.
- **Supported roles**:
  - `"human"` / `"user"` → `HumanMessage`
  - `"ai"` / `"assistant"` → `AIMessage`
  - `"system"` → `SystemMessage`
  - `"tool"` → `ToolMessage` (requires `tool_call_id` and `name`)
- **Why tool messages?**: Historical tool results are needed for context (e.g., "You previously calculated 6, now multiply that by...").

### `convert_from_langchain_messages(messages: List) -> List[Dict[str, Any]]`
**Purpose**: Translates LangChain messages to JSON for API responses.
- **Flow**: Iterates through messages → checks instance type → builds dict.
- **Special handling for AIMessages with tool_calls**:
  - Extracts `tool_calls` list and formats it
  - Preserves the structure for client-side rendering
- **Special handling for ToolMessages**:
  - Extracts `tool_call_id` to link result back to the request
  - Extracts `name` for display purposes
- **Why preserve tool_calls?**: The client might want to display "Assistant is requesting to use multiply(2, 3)" before approval.

---

## 🚀 MODULE 7: FASTAPI APPLICATION (Lifespan & Core)

### Lifespan Manager
**`@asynccontextmanager async def lifespan(app: FastAPI):`**
- **Purpose**: Manages resources that live for the entire application lifetime.
- **Enter phase (server startup)**:
  1. Opens SQLite connection: `AsyncSqliteSaver.from_conn_string("breakpoints_memory.db")`
  2. Builds graph (no compilation yet)
  3. **Compiles with interrupt**: `builder.compile(interrupt_before=["tools"], checkpointer=checkpointer)` — this is where the magic happens! The graph will pause *before* executing the tools node.
  4. Stores graph in `app.state.graph` for endpoints to access
  5. Initializes `app.state.pending_approvals` dict for tracking active approval requests
- **`yield`**: Server runs here. All requests are processed while in this state.
- **Exit phase (server shutdown)**: Cleanup happens here (currently just a print, but could close other resources).

### FastAPI App Instance
**`app = FastAPI(...)`**
- **Title & Description**: Appear in Swagger UI (`/docs`)
- **Version**: API version tracking
- **Lifespan**: Attaches the lifespan manager

### State Storage
- **`app.state.graph`**: The compiled LangGraph. Shared across all requests. Thread-safe (each request uses its own thread_id).
- **`app.state.pending_approvals`**: Dictionary:
  - Key: `thread_id` (string)
  - Value: `{"expires_at": datetime, "tool_call": {...}}`
- **Why in-memory?**: Pending approvals are ephemeral. If the server restarts, pending approvals expire naturally.

---

## 🌐 MODULE 8: ENDPOINTS

### GET `/`
**Purpose**: API root — provides service information and endpoint documentation.
- Returns a JSON object listing all available endpoints.
- Helpful for discovery and quick testing.

### POST `/process` (Core Endpoint)
**Purpose**: Main entry point for user messages.
- **Input**: `MessageRequest` (messages + thread_id)
- **Process**:
  1. Converts incoming JSON to LangChain messages
  2. Creates initial input: `{"messages": langchain_messages}`
  3. Configures thread: `{"configurable": {"thread_id": request.thread_id}}`
  4. **Streams graph execution**: `graph.astream()` sends events as they happen
  5. **Detects interruption**: Checks if the last message was an AI message with `tool_calls`
  6. **If interrupted**: Stores pending approval with 5-minute TTL, returns `pending_approval` status
  7. **If complete**: Returns final messages with `complete` status
- **Error handling**: Catches all exceptions and returns 500 with descriptive message.

### POST `/approve`
**Purpose**: Handles user approval/rejection of tool calls.
- **Input**: `ApprovalRequest` (thread_id + approved boolean)
- **Process**:
  1. Checks if there's a pending approval for this thread
  2. Validates it hasn't expired (5-minute TTL)
  3. **If rejected**: Deletes pending entry, returns cancellation confirmation
  4. **If approved**: 
     - Streams graph with `None` input (resumes from interruption)
     - Graph executes the tool node automatically
     - Returns final result with `executed` status
- **Important**: After resuming, the graph completes the tool call and returns to the assistant for the final answer.

### GET `/state/{thread_id}`
**Purpose**: Inspect full conversation state (debugging/audit).
- **Process**:
  1. `await graph.aget_state(thread_config)` — retrieves the current state snapshot from SQLite
  2. Checks if state exists (`state.values`)
  3. Returns:
     - `exists`: Boolean
     - `next_nodes`: Which nodes are scheduled to run next (useful for understanding graph position)
     - `message_count`: How many messages in history
     - `messages`: Full message list (converted to JSON)
     - `pending_approval`: Whether there's an active approval request

### GET `/pending/{thread_id}`
**Purpose**: Lightweight check for pending approvals (without fetching full state).
- **Process**:
  1. Checks the `pending_approvals` dict
  2. If expired: cleans up the entry
  3. Returns:
     - `pending`: Boolean
     - `expires_at`: ISO timestamp (if pending)
     - `tool_call`: Tool details (if pending)

### GET `/health`
**Purpose**: Service health check for load balancers/monitoring.
- Returns simple status with model and tool info.
- Useful for container orchestration (Kubernetes liveness/readiness probes).

---

## 🔄 MODULE 9: MAIN ENTRYPOINT

### `if __name__ == "__main__":`
**Purpose**: Runs the server when the script is executed directly (not imported).
- **`import uvicorn`**: ASGI server for FastAPI.
- **`uvicorn.run(...)`**: Starts the server:
  - `"M3-LG-braekingPoints:app"`: Module name (your filename) and the FastAPI instance name (`app`)
  - `host="0.0.0.0"`: Listens on all network interfaces (accessible from other machines)
  - `port=8000`: Standard development port
  - `reload=True`: Auto-restarts on code changes (development only — disable in production)

---

## 🎯 KEY CONCEPTS SUMMARY

### The Interrupt Pattern
- **`interrupt_before=["tools"]`** tells LangGraph to pause before executing the tools node.
- The graph stops with the AI message containing `tool_calls` in the state.
- User approval is received via the `/approve` endpoint.
- Passing `None` as input to `astream` resumes from the interruption point.

### Checkpointing (Persistence)
- **`AsyncSqliteSaver`** saves the entire graph state to `breakpoints_memory.db`.
- Each `thread_id` gets its own checkpoint "folder" within the database.
- State persists across server restarts, but pending approvals are in-memory (intentional).

### State Reducers
- LangGraph uses reducers to handle state updates. When you return `{"messages": [new_msg]}`, the default reducer appends to the existing list.
- `RemoveMessage` (not used here but in previous example) can delete messages from state.

### Human-in-the-Loop Architecture
- **Synchronous**: User request → graph pauses → API returns pending → user acts → graph resumes.
- **Benefits**: 
  - Users can review tool calls before execution
  - Allows for rejection/correction
  - Prevents unauthorized or costly operations

---

## 🏗️ PRODUCTION RECOMMENDATIONS

### Security
- **Rate limiting**: Add middleware to prevent abuse (e.g., 100 requests/minute per thread_id)
- **Authentication**: Add API keys or JWT for endpoints
- **Validation**: Stricter Pydantic validators for message content (length limits, sanitization)

### Performance
- **Connection pooling**: `AsyncSqliteSaver` can be reused instead of creating new connections
- **Caching**: Cache frequently accessed state (e.g., last 10 messages) in Redis
- **Background tasks**: For long-running operations, use Celery + Redis queue

### Monitoring
- **Structured logging**: Add `loguru` or `structlog` for JSON logs
- **Tracing**: Integrate LangSmith for LLM call tracing
- **Metrics**: Add Prometheus metrics for request latency, token usage, error rates

### Reliability
- **Timeouts**: Set timeouts on LLM calls (already done via `ChatOpenAI` defaults)
- **Retries**: Configure `ChatOpenAI` with `max_retries=3`
- **Graceful shutdown**: Ensure checkpointer closes connections properly

---

This architecture document covers every module, function, class, and concept in the code. You can paste the explanations directly into your file as comments or use them for documentation.