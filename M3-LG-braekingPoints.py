"""
FastAPI adaptation of LangGraph Breakpoints (Human-in-the-loop)
Module 3, Lesson 2: User approval before tool execution
"""

import asyncio
import json
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# ============================================================================
# Configuration with fallbacks and validation
# ============================================================================
load_dotenv()

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")


# ============================================================================
# Tools Definition
# ============================================================================
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b

def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a by b."""
    return a / b

tools = [add, multiply, divide]

# ============================================================================
# LLM and Graph Setup
# ============================================================================
llm = ChatOpenAI(
    model=DEFAULT_MODEL,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE,
    temperature=0.0,
)
llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(
    content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
)

def assistant(state: MessagesState):
    """Assistant node - calls LLM with tools."""
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

def build_graph():
    """Build the LangGraph with breakpoint before tools."""
    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")
    
    return builder

# ============================================================================
# Pydantic Models
# ============================================================================
class MessageRequest(BaseModel):
    messages: List[Dict[str, str]] = Field(..., description="List of messages with 'role' and 'content'")
    thread_id: str = Field(..., description="Unique conversation thread ID")

class ApprovalRequest(BaseModel):
    thread_id: str = Field(..., description="Thread ID to approve")
    approved: bool = Field(..., description="True to execute tool, False to cancel")

class ProcessResponse(BaseModel):
    status: str = Field(..., description="Status: 'complete' or 'pending_approval'")
    messages: Optional[List[Dict[str, Any]]] = Field(None, description="Response messages")
    tool_call: Optional[Dict[str, Any]] = Field(None, description="Tool call awaiting approval")
    thread_id: str = Field(..., description="Thread ID")

# ============================================================================
# Message Helpers
# ============================================================================
def convert_to_langchain_messages(messages: List[Dict[str, str]]) -> List:
    """Convert API dict messages to LangChain message objects."""
    langchain_messages = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "human" or role == "user":
            langchain_messages.append(HumanMessage(content=content))
        elif role == "ai" or role == "assistant":
            langchain_messages.append(AIMessage(content=content))
        elif role == "system":
            langchain_messages.append(SystemMessage(content=content))
        elif role == "tool":
            langchain_messages.append(ToolMessage(
                content=content,
                tool_call_id=msg.get("tool_call_id", ""),
                name=msg.get("name", "")
            ))
    return langchain_messages

def convert_from_langchain_messages(messages: List) -> List[Dict[str, Any]]:
    """Convert LangChain message objects to API dict format."""
    result = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            result.append({"role": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                result.append({
                    "role": "ai",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.get("id"),
                            "name": tc.get("name"),
                            "args": tc.get("args"),
                        }
                        for tc in msg.tool_calls
                    ]
                })
            else:
                result.append({"role": "ai", "content": msg.content})
        elif isinstance(msg, ToolMessage):
            result.append({
                "role": "tool",
                "content": msg.content,
                "tool_call_id": msg.tool_call_id,
                "name": msg.name,
            })
        elif isinstance(msg, SystemMessage):
            result.append({"role": "system", "content": msg.content})
    return result

# ============================================================================
# FastAPI Application
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for database connections."""
    async with AsyncSqliteSaver.from_conn_string("breakpoints_memory.db") as checkpointer:
        builder = build_graph()
        graph = builder.compile(
            interrupt_before=["tools"],
            checkpointer=checkpointer
        )
        app.state.graph = graph
        app.state.pending_approvals = {}
        
        print("✅ Graph compiled with breakpoint before tools")
        print("✅ SQLite checkpointer initialized")
        
        yield
        
        print("🔄 Cleaning up...")

app = FastAPI(
    title="LangGraph Breakpoints API",
    description="Human-in-the-loop with tool approval",
    version="1.0.0",
    lifespan=lifespan,
)

# ============================================================================
# Endpoints
# ============================================================================

@app.get("/")
async def root():
    return {
        "service": "LangGraph Breakpoints API",
        "endpoints": {
            "POST /process": "Send messages (may require approval)",
            "POST /approve": "Approve or reject pending tool call",
            "GET /state/{thread_id}": "Get conversation state",
            "GET /pending/{thread_id}": "Check if thread has pending approval",
        }
    }

@app.post("/process", response_model=ProcessResponse)
async def process_messages(request: MessageRequest):
    """Process messages through the LangGraph."""
    graph = app.state.graph
    langchain_messages = convert_to_langchain_messages(request.messages)
    initial_input = {"messages": langchain_messages}
    thread_config = {"configurable": {"thread_id": request.thread_id}}
    
    try:
        state_snapshot = None
        interrupted = False
        tool_call_info = None
        
        async for event in graph.astream(
            initial_input,
            thread_config,
            stream_mode="values"
        ):
            state_snapshot = event
            messages = event.get("messages", [])
            if messages:
                last_msg = messages[-1]
                if isinstance(last_msg, AIMessage) and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    interrupted = True
                    tool_call_info = {
                        "id": last_msg.tool_calls[0].get("id"),
                        "name": last_msg.tool_calls[0].get("name"),
                        "args": last_msg.tool_calls[0].get("args"),
                    }
        
        if interrupted and tool_call_info:
            app.state.pending_approvals[request.thread_id] = {
                "expires_at": datetime.now() + timedelta(minutes=5),
                "tool_call": tool_call_info,
            }
            return ProcessResponse(
                status="pending_approval",
                messages=convert_from_langchain_messages(state_snapshot.get("messages", [])),
                tool_call=tool_call_info,
                thread_id=request.thread_id,
            )
        else:
            return ProcessResponse(
                status="complete",
                messages=convert_from_langchain_messages(state_snapshot.get("messages", [])) if state_snapshot else [],
                thread_id=request.thread_id,
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {str(e)}")

@app.post("/approve")
async def approve_tool(approval: ApprovalRequest):
    """Approve or reject a pending tool call."""
    graph = app.state.graph
    thread_id = approval.thread_id
    thread_config = {"configurable": {"thread_id": thread_id}}
    
    pending = app.state.pending_approvals.get(thread_id)
    if not pending:
        raise HTTPException(status_code=404, detail="No pending approval found for this thread")
    
    if datetime.now() > pending["expires_at"]:
        del app.state.pending_approvals[thread_id]
        raise HTTPException(status_code=410, detail="Approval request has expired")
    
    if not approval.approved:
        del app.state.pending_approvals[thread_id]
        return {
            "status": "cancelled",
            "thread_id": thread_id,
            "message": "Tool execution cancelled by user"
        }
    
    try:
        final_state = None
        async for event in graph.astream(
            None,
            thread_config,
            stream_mode="values"
        ):
            final_state = event
        
        del app.state.pending_approvals[thread_id]
        
        return {
            "status": "executed",
            "thread_id": thread_id,
            "messages": convert_from_langchain_messages(final_state.get("messages", [])) if final_state else [],
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {str(e)}")

@app.get("/state/{thread_id}")
async def get_state(thread_id: str):
    """Get the current state of a conversation thread."""
    graph = app.state.graph
    thread_config = {"configurable": {"thread_id": thread_id}}
    
    try:
        state = await graph.aget_state(thread_config)
        if not state or not state.values:
            return {"thread_id": thread_id, "exists": False}
        
        messages = state.values.get("messages", [])
        return {
            "thread_id": thread_id,
            "exists": True,
            "next_nodes": list(state.next) if state.next else [],
            "message_count": len(messages),
            "messages": convert_from_langchain_messages(messages),
            "pending_approval": thread_id in app.state.pending_approvals,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get state: {str(e)}")

@app.get("/pending/{thread_id}")
async def check_pending(thread_id: str):
    """Check if a thread has a pending approval."""
    pending = app.state.pending_approvals.get(thread_id)
    if not pending:
        return {"thread_id": thread_id, "pending": False}
    
    expired = datetime.now() > pending["expires_at"]
    if expired:
        del app.state.pending_approvals[thread_id]
        return {"thread_id": thread_id, "pending": False}
    
    return {
        "thread_id": thread_id,
        "pending": True,
        "expires_at": pending["expires_at"].isoformat(),
        "tool_call": pending["tool_call"],
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": DEFAULT_MODEL,
        "tools": [t.__name__ for t in tools],
    }

# ============================================================================
# Main Entrypoint - CORRECTED for your filename
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "M3-LG-braekingPoints:app",  # ← Matches your filename exactly
        host="0.0.0.0",
        port=8000,
        reload=True,
    )