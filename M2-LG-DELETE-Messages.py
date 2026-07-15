import os
from contextlib import asynccontextmanager
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# =============== Env Setup ===============
# Uses the exact same .env file and settings as before
load_dotenv()

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

llm = ChatOpenAI(
    model=DEFAULT_MODEL,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE,
    temperature=0.0,
)

# =============== Graph Nodes ===============
async def filter_messages(state: MessagesState):
    """
    THE BEST EXAMPLE: Reducer with RemoveMessage.
    Permanently deletes all but the 2 most recent messages from the state/database.
    This prevents the SQLite DB from growing infinitely and keeps the context window clean.
    """
    messages = state["messages"]
    
    # Only delete if we have more than 2 messages
    if len(messages) > 1:
        # Create RemoveMessage objects for all messages except the last 2
        # LangGraph's built-in reducer will see these and physically delete them from the DB
        delete_messages = [RemoveMessage(id=m.id) for m in messages[:-1]]
        return {"messages": delete_messages}
    
    return {"messages": []}

async def call_model(state: MessagesState):
    """Calls the LLM with the filtered messages."""
    # Because the previous node already deleted the old messages, 
    # the LLM only sees the 2 most recent messages.
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}

# =============== FastAPI Lifespan & Graph Compilation ===============
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Use AsyncSqliteSaver for persistent, non-blocking memory
    # We use a new DB file name to keep this experiment separate
    async with AsyncSqliteSaver.from_conn_string("trim_filter_memory.db") as memory:
        builder = StateGraph(MessagesState)
        
        # Add nodes
        builder.add_node("filter", filter_messages)
        builder.add_node("chat_model", call_model)
        
        # Define edges: START -> filter -> chat_model -> END
        builder.add_edge(START, "filter")
        builder.add_edge("filter", "chat_model")
        builder.add_edge("chat_model", END)
        
        # Compile and attach to app.state
        app.state.trim_graph = builder.compile(checkpointer=memory)
        yield

app = FastAPI(title="LangGraph Trim/Filter Messages API", lifespan=lifespan)

# =============== Models ===============
class MessageRequest(BaseModel):
    messages: List[dict]
    thread_id: str

class MessageResponse(BaseModel):
    messages: List[dict]
    response: str

# =============== Helper Functions ===============
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

# =============== Endpoints ===============
@app.get("/")
async def root():
    return {
        "message": "LangGraph Trim/Filter Messages API", 
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
        
        return MessageResponse(
            messages=all_messages,
            response=final_response
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state/{thread_id}")
async def get_state(thread_id: str):
    """
    Utility endpoint to inspect the SQLite DB.
    Use this to prove that old messages are being permanently deleted!
    """
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

# =============== Running the app ===============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)