from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware
from langchain.messages import HumanMessage, AIMessage, ToolMessage, RemoveMessage
from langchain.agents.middleware import before_agent
from langgraph.runtime import Runtime
from langchain.agents import AgentState
from typing import Any
from langchain_openai import ChatOpenAI

load_dotenv()

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

llm = ChatOpenAI(
    model=DEFAULT_MODEL,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE,
)

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
                model=llm,
                trigger=("tokens", 100),
                keep=("messages", 1)
            )
        )
    
    if use_trimming:
        middleware.append(trim_messages)
    
    return create_agent(
        model=llm,
        checkpointer=InMemorySaver(),
        middleware=middleware
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