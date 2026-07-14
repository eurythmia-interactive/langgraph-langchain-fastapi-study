import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# =============== Env Setup ===============
load_dotenv()

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

llm = ChatOpenAI(
    model=DEFAULT_MODEL,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE,
)

# =============== Tools ===============
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools = [add, multiply, divide]
llm_with_tools = llm.bind_tools(tools)

# =============== Graph Nodes ===============
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# =============== Graph Factory & Compilation ===============
# CRITICAL: For memory to persist across API requests, the checkpointer and compiled graph 
# must be instantiated once globally at startup, not per request.
memory = MemorySaver()

builder = StateGraph(MessagesState)

# Define nodes
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition, # Routes to 'tools' if tool call, else to END
)
builder.add_edge("tools", "assistant")

# Compile with checkpointer
react_graph_memory = builder.compile(checkpointer=memory)

# =============== FastAPI App ===============
app = FastAPI(title="LangGraph Agent Memory API")

# =============== Models ===============
class MessageRequest(BaseModel):
    messages: List[dict]  # Will contain role and content
    thread_id: str

class MessageResponse(BaseModel):
    messages: List[dict]
    response: str

# =============== Helper Functions ===============
def convert_to_langchain_messages(messages: List[dict]) -> List:
    """Convert dict messages to LangChain message objects"""
    langchain_messages = []
    for msg in messages:
        role = msg.get("role", "").lower()
        content = msg.get("content", "")
        
        if role in ["human", "user"]:
            langchain_messages.append(HumanMessage(content=content))
        elif role in ["ai", "assistant"]:
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
        elif isinstance(msg, SystemMessage):
            result.append({"role": "system", "content": msg.content})
    return result

# =============== Endpoints ===============
@app.get("/")
async def root():
    return {
        "message": "LangGraph Agent Memory API", 
        "endpoints": ["/process"]
    }

@app.post("/process", response_model=MessageResponse)
async def process_messages(request: MessageRequest):
    """
    Process messages with persistent memory using thread_id.
    Example message format:
    [
        {"role": "human", "content": "Add 3 and 4."},
        {"role": "human", "content": "Multiply that by 2."}
    ]
    """
    try:
        # Convert messages to LangChain format
        langchain_messages = convert_to_langchain_messages(request.messages)
        
        # Configure the thread for memory persistence
        config = {"configurable": {"thread_id": request.thread_id}}
        
        # Invoke the graph asynchronously (prevents blocking the FastAPI event loop)
        response = await react_graph_memory.ainvoke(
            {"messages": langchain_messages}, 
            config
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

# =============== Running the app ===============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)