"""
=============================================================================
LangGraph Token-Based Message Trimming API
=============================================================================
This application demonstrates how to use LangChain's `trim_messages` utility 
to restrict the conversation history sent to the LLM based on a strict token limit.
=============================================================================
"""

import os
from contextlib import asynccontextmanager
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- LangChain / LangGraph Imports ---
from langchain_openai import ChatOpenAI
# We import trim_messages, which is a utility function to slice message lists by token count
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, trim_messages
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# =============================================================================
# 1. ENVIRONMENT & LLM SETUP
# =============================================================================
load_dotenv()

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

# We initialize the LLM globally. 
# CRITICAL: We will use this exact `llm` object as our `token_counter` later!
llm = ChatOpenAI(
    model=DEFAULT_MODEL,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE,
    temperature=0.0,
)

# =============================================================================
# 2. LANGGRAPH NODES (The "Workers")
# =============================================================================
async def chat_model_node(state: MessagesState):
    """
    This node trims the message history to a strict token limit before 
    passing it to the LLM. 
    """
    # 1. TRIM THE MESSAGES
    # trim_messages is a LangChain utility that calculates token counts and slices the list.
    trimmed_messages = trim_messages(
        state["messages"],      # The full list of messages from the state
        max_tokens=100,         # The strict token limit (set low for demonstration)
        strategy="last",        # "last" means "keep the most recent messages"
        token_counter=llm,      # We pass our LLM instance! It knows how to count its own tokens.
        allow_partial=False,    # If a message doesn't fit, exclude it entirely (don't cut it in half)
    )
    
    # 2. CALL THE LLM
    # We only pass the `trimmed_messages` to the AI, saving context window and money!
    response = await llm.ainvoke(trimmed_messages)
    
    # 3. RETURN THE RESULT
    # LangGraph will append this new AI response to the full, un-trimmed state history.
    return {"messages": [response]}

# =============================================================================
# 3. FASTAPI LIFESPAN & GRAPH COMPILATION
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Open the async SQLite connection for persistent memory
    async with AsyncSqliteSaver.from_conn_string("token_trim_memory.db") as memory:
        builder = StateGraph(MessagesState)
        
        # Add our single node
        builder.add_node("chat_model", chat_model_node)
        
        # Define the simple flow: START -> chat_model -> END
        builder.add_edge(START, "chat_model")
        builder.add_edge("chat_model", END)
        
        # Compile with the SQLite checkpointer and attach to app.state
        app.state.trim_graph = builder.compile(checkpointer=memory)
        yield

app = FastAPI(title="LangGraph Token Trimming API", lifespan=lifespan)

# =============================================================================
# 4. PYDANTIC MODELS (The "Bouncers")
# =============================================================================
class MessageRequest(BaseModel):
    messages: List[dict]
    thread_id: str

class MessageResponse(BaseModel):
    messages: List[dict]
    response: str

# =============================================================================
# 5. HELPER FUNCTIONS (The "Translators")
# =============================================================================
def convert_to_langchain_messages(messages: List[dict]) -> List:
    """Converts incoming JSON dictionaries into LangChain Message objects."""
    langchain_messages = []
    for msg in messages:
        role = msg.get("role", "").lower()
        content = msg.get("content", "")
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
# 6. FASTAPI ENDPOINTS (The "Waiters")
# =============================================================================
@app.get("/")
async def root():
    return {
        "message": "LangGraph Token Trimming API", 
        "endpoints": ["/process", "/state/{thread_id}"]
    }

@app.post("/process", response_model=MessageResponse)
async def process_messages(request: MessageRequest):
    try:
        langchain_messages = convert_to_langchain_messages(request.messages)
        config = {"configurable": {"thread_id": request.thread_id}}
        
        # Run the graph asynchronously
        response = await app.state.trim_graph.ainvoke(
            {"messages": langchain_messages}, 
            config
        )
        
        final_message = response["messages"][-1]
        final_response = final_message.content if hasattr(final_message, 'content') else str(final_message)
        all_messages = convert_from_langchain_messages(response["messages"])
        
        return MessageResponse(messages=all_messages, response=final_response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state/{thread_id}")
async def get_state(thread_id: str):
    """Utility endpoint to inspect the raw state saved in the SQLite DB."""
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = await app.state.trim_graph.aget_state(config)
        
        if not state.values:
            return {"message": "No state found for this thread_id"}
        
        messages = convert_from_langchain_messages(state.values.get("messages", []))
        
        return {
            "thread_id": thread_id,
            "message_count": len(messages),
            "messages": messages
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# 7. RUNNING THE APP
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)