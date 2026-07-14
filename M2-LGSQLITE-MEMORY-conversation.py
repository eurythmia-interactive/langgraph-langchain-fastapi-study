"""
=============================================================================
LangGraph Chatbot with External SQLite Memory & FastAPI
=============================================================================
This application demonstrates how to build a stateful AI chatbot that:
1. Uses LangGraph to manage conversation flow and state.
2. Summarizes old messages to prevent context window overflow.
3. Persists memory to an external SQLite database so it survives server restarts.
4. Serves the AI via a high-performance, asynchronous FastAPI web server.
=============================================================================
"""

import os
from contextlib import asynccontextmanager
from typing import List, Literal
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- LangChain / LangGraph Imports ---
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# =============================================================================
# 1. ENVIRONMENT & LLM SETUP
# =============================================================================
# Load environment variables from the .env file (API keys, model names, etc.)
load_dotenv()

# Fetch configuration from environment variables, with safe fallback defaults
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

# Initialize the Language Model (LLM). 
# We set temperature=0.0 for deterministic, logical responses.
# This instance is created ONCE at startup and reused for all requests.
llm = ChatOpenAI(
    model=DEFAULT_MODEL,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE,
    temperature=0.0,
)

# =============================================================================
# 2. LANGGRAPH STATE DEFINITION (The "Backpack")
# =============================================================================
# In LangGraph, the "State" is a dictionary-like object passed between nodes.
# We inherit from MessagesState, which is a pre-built state that already includes 
# a "messages" list and knows how to automatically append new messages to it.
# We add a custom "summary" field to store the compressed conversation history.
class State(MessagesState):
    summary: str

# =============================================================================
# 3. LANGGRAPH NODES (The "Workers")
# =============================================================================
# Nodes are just Python functions that read from the State, do some work, 
# and return updates to the State. We make them `async` so they don't block 
# the FastAPI server while waiting for the LLM to respond.

async def call_model(state: State):
    """
    Node 1: The Conversationalist.
    Reads the current state, injects the summary (if it exists) as a secret 
    system message, and asks the LLM for a response.
    """
    # 1. Check if we have a summary from previous turns
    summary = state.get("summary", "")
    
    # 2. If a summary exists, prepend it to the messages as a SystemMessage.
    # This gives the LLM "long-term memory" without using up the context window.
    if summary:
        system_message = f"Summary of conversation earlier: {summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        # If no summary, just pass the raw messages
        messages = state["messages"]
    
    # 3. Call the LLM asynchronously (ainvoke) to get the AI's response
    response = await llm.ainvoke(messages)
    
    # 4. Return the AI's response. LangGraph's built-in reducer will automatically 
    # append this to the "messages" list in the State.
    return {"messages": [response]}

async def summarize_conversation(state: State):
    """
    Node 2: The Note-Taker.
    Triggered when the conversation gets too long. It asks the LLM to summarize 
    the chat, updates the summary in the state, and deletes old messages to 
    save space (context window management).
    """
    summary = state.get("summary", "")
    
    # 1. Create the prompt for the LLM. If a summary already exists, we ask 
    # the LLM to *extend* it. Otherwise, we ask it to create a new one.
    if summary:
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above: "
        )
    else:
        summary_message = "Create a summary of the conversation above: "

    # 2. Append the summarization instruction as a HumanMessage to the chat history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    
    # 3. Ask the LLM to generate the summary
    response = await llm.ainvoke(messages)
    
    # 4. CRITICAL STEP: Delete old messages to prevent the context window from overflowing.
    # We keep only the 2 most recent messages (usually the last Human/AI pair).
    # RemoveMessage tells LangGraph's reducer to delete these specific messages by ID.
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    
    # 5. Return the new summary and the list of messages to delete
    return {"summary": response.content, "messages": delete_messages}

# =============================================================================
# 4. LANGGRAPH EDGES (The "Traffic Cops")
# =============================================================================
def should_continue(state: State) -> Literal["summarize_conversation", END]:
    """
    Conditional Edge: Decides where the graph should go next after the 
    'conversation' node finishes.
    """
    messages = state["messages"]
    
    # If the conversation history exceeds 6 messages, route to the summarizer
    if len(messages) > 6:
        return "summarize_conversation"
    
    # Otherwise, the conversation is short enough, so we just end the graph run
    return END

# =============================================================================
# 5. FASTAPI LIFESPAN & GRAPH COMPILATION (The "Store Manager")
# =============================================================================
# FastAPI's `lifespan` context manager allows us to run setup code when the 
# server starts, and teardown code when the server shuts down.
# This is REQUIRED for AsyncSqliteSaver because it needs to open and close 
# the database connection pool safely.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP PHASE ---
    # Open the async SQLite connection. "chatbot_memory.db" is the file on disk.
    async with AsyncSqliteSaver.from_conn_string("chatbot_memory.db") as memory:
        
        # Build the LangGraph StateGraph using our State definition
        builder = StateGraph(State)
        
        # Register our nodes (workers)
        builder.add_node("conversation", call_model)
        builder.add_node("summarize_conversation", summarize_conversation)
        
        # Connect the nodes with edges (paths)
        builder.add_edge(START, "conversation") # Always start at the conversation node
        builder.add_conditional_edges("conversation", should_continue) # Ask the traffic cop
        builder.add_edge("summarize_conversation", END) # After summarizing, end the run
        
        # Compile the graph with the SQLite memory attached.
        # We attach the compiled graph to `app.state` so our endpoints can access it!
        app.state.chatbot_graph = builder.compile(checkpointer=memory)
        
        # The `yield` pauses the lifespan function and hands control back to FastAPI 
        # to start accepting web requests. The graph and DB connection are now live.
        yield
        
    # --- SHUTDOWN PHASE ---
    # When you stop the server (e.g., Ctrl+C), execution resumes here.
    # The `async with` block automatically and safely closes the SQLite connection.

# Initialize the FastAPI app and pass in our lifespan manager
app = FastAPI(title="LangGraph Chatbot with External Memory API", lifespan=lifespan)

# =============================================================================
# 6. PYDANTIC MODELS (The "Bouncers")
# =============================================================================
# Pydantic models define the exact shape of the JSON data we expect to receive 
# and send over the internet. They validate incoming requests automatically.

class MessageRequest(BaseModel):
    messages: List[dict]  # Expect a list of dictionaries (e.g., [{"role": "human", "content": "Hi"}])
    thread_id: str        # Expect a string to identify the conversation thread for memory

class MessageResponse(BaseModel):
    messages: List[dict]  # The full updated conversation history
    response: str         # The final AI response text
    summary: str          # The current running summary (if one was generated)

# =============================================================================
# 7. HELPER FUNCTIONS (The "Translators")
# =============================================================================
# The internet speaks in simple JSON dictionaries. LangGraph speaks in complex 
# Python objects (HumanMessage, AIMessage, etc.). These functions translate between them.

def convert_to_langchain_messages(messages: List[dict]) -> List:
    """Converts incoming JSON dictionaries into LangChain Message objects."""
    langchain_messages = []
    for msg in messages:
        role = msg.get("role", "").lower()
        content = msg.get("content", "")
        
        # Map the string roles to the specific LangChain classes
        if role in ["human", "user"]:
            langchain_messages.append(HumanMessage(content=content))
        elif role in ["ai", "assistant"]:
            langchain_messages.append(AIMessage(content=content))
        elif role == "tool":
            langchain_messages.append(ToolMessage(content=content, tool_call_id=msg.get("tool_call_id", "1")))
    return langchain_messages

def convert_from_langchain_messages(messages: List) -> List[dict]:
    """Converts LangChain Message objects back into simple JSON dictionaries."""
    result = []
    for msg in messages:
        # Check the type of the message and convert it to a dict
        if isinstance(msg, HumanMessage):
            result.append({"role": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "ai", "content": msg.content})
        elif isinstance(msg, ToolMessage):
            result.append({"role": "tool", "content": msg.content})
        elif isinstance(msg, SystemMessage):
            result.append({"role": "system", "content": msg.content})
    return result

# =============================================================================
# 8. FASTAPI ENDPOINTS (The "Waiters")
# =============================================================================
# These are the actual URLs that your frontend or Postman will call.

@app.get("/")
async def root():
    """A simple health-check and documentation endpoint."""
    return {
        "message": "LangGraph Chatbot with External Memory API", 
        "endpoints": ["/process", "/state/{thread_id}"]
    }

@app.post("/process", response_model=MessageResponse)
async def process_messages(request: MessageRequest):
    """
    Main endpoint: Receives user messages, runs them through the LangGraph, 
    and returns the AI response along with the updated state.
    """
    try:
        # 1. Translate JSON dicts to LangChain objects
        langchain_messages = convert_to_langchain_messages(request.messages)
        
        # 2. Create the config dictionary. The `thread_id` is the magic key that 
        # tells the SQLite checkpointer which conversation folder to load/save.
        config = {"configurable": {"thread_id": request.thread_id}}
        
        # 3. Run the graph! We use `ainvoke` (async invoke) so the server doesn't freeze.
        # We pass the messages and the config. LangGraph handles the rest.
        response = await app.state.chatbot_graph.ainvoke(
            {"messages": langchain_messages}, 
            config
        )
        
        # 4. Extract the final AI message and the summary from the resulting state
        final_message = response["messages"][-1]
        final_response = final_message.content if hasattr(final_message, 'content') else str(final_message)
        summary = response.get("summary", "")
        
        # 5. Translate the LangChain objects back to JSON dicts for the user
        all_messages = convert_from_langchain_messages(response["messages"])
        
        # 6. Return the structured response
        return MessageResponse(
            messages=all_messages,
            response=final_response,
            summary=summary
        )
    except Exception as e:
        # Catch any errors and return a standard HTTP 500 response
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state/{thread_id}")
async def get_state(thread_id: str):
    """
    Utility endpoint: Allows you to peek directly into the SQLite database 
    to see exactly what LangGraph has saved for a specific thread_id.
    Great for debugging and proving that memory persists!
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        
        # Use `aget_state` to fetch the current state from the checkpointer without running the graph
        state = await app.state.chatbot_graph.aget_state(config)
        
        # If the thread_id doesn't exist in the DB yet, return a friendly message
        if not state.values:
            return {"message": "No state found for this thread_id"}
        
        # Extract the saved summary and messages
        summary = state.values.get("summary", "")
        messages = convert_from_langchain_messages(state.values.get("messages", []))
        
        return {
            "thread_id": thread_id,
            "summary": summary,
            "message_count": len(messages),
            "messages": messages
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# 9. RUNNING THE APP
# =============================================================================
# Standard Python boilerplate to run the file directly via `python script_name.py`
if __name__ == "__main__":
    import uvicorn
    # Start the Uvicorn ASGI server on port 8000, accessible from any IP (0.0.0.0)
    uvicorn.run(app, host="0.0.0.0", port=8000)