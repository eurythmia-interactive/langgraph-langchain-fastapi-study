from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware
from langchain.messages import HumanMessage, AIMessage, SystemMessage
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

checkpointer = InMemorySaver()

app = FastAPI(title="AI Engineering Assistant API")

SYSTEM_PROMPT = """You are an AI engineering assistant specialized in:
- AI/ML fundamentals and theory
- Deep learning and neural networks
- Large Language Models (LLMs) and transformers
- LangChain, LangGraph, and agent frameworks
- Prompt engineering techniques
- RAG (Retrieval-Augmented Generation) systems
- Vector databases and embeddings
- Model training, fine-tuning, and deployment

IMPORTANT: If asked about topics outside AI engineering (such as cooking, sports, politics, general knowledge, etc.), you must respond with:
"I'm sorry, I can only help with AI engineering questions. I don't have expertise in other subjects."

Always be helpful, technical, and focused on AI engineering topics."""

GREETING = "Hello! I'm your AI engineering assistant. How can I help you today with AI, machine learning, or related technologies?"

# =============== Models ===============

class ChatRequest(BaseModel):
    message: str
    thread_id: str

class ChatResponse(BaseModel):
    response: str
    thread_id: str
    conversation_history: List[dict]
    conversation_ended: bool

# =============== Agent Factory ===============

def create_chat_agent():
    """Create agent with summarization middleware"""
    middleware = [
        SummarizationMiddleware(
            model=llm,
            trigger=("tokens", 100),
            keep=("messages", 2),
            trim_tokens_to_summarize=500
        )
    ]
    
    return create_agent(
        model=llm,
        checkpointer=checkpointer,
        middleware=middleware,
        system_prompt=SYSTEM_PROMPT
    )

# =============== Helper Functions ===============

def extract_summary(messages: List) -> str:
    """Extract summary from message history"""
    for msg in reversed(messages):
        if hasattr(msg, 'content') and isinstance(msg.content, str):
            if msg.content.startswith("Summary:") or "summary" in msg.content.lower():
                return msg.content
    return ""

def convert_messages_to_dict(messages: List) -> List[dict]:
    """Convert LangChain messages to dict format"""
    result = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            result.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "assistant", "content": msg.content})
    return result

# =============== Endpoints ===============

@app.get("/")
async def root():
    return {
        "message": "AI Engineering Assistant API",
        "endpoints": ["/chat"],
        "description": "Conversational AI assistant specialized in AI engineering topics"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for AI engineering conversation
    
    - First message with a thread_id starts a new conversation
    - Subsequent messages with same thread_id continue the conversation
    - Send "exit" to end the conversation
    """
    try:
        # Handle exit command
        if request.message.strip().lower() == "exit":
            return ChatResponse(
                response="Goodbye! Thanks for chatting about AI engineering.",
                thread_id=request.thread_id,
                conversation_history=[],
                conversation_ended=True
            )
        
        # Create agent
        agent = create_chat_agent()
        
        # Check if this is a new conversation by trying to get state
        try:
            state = agent.get_state({"configurable": {"thread_id": request.thread_id}})
            is_new_conversation = state is None or not state.values.get("messages")
        except:
            is_new_conversation = True
        
        # Prepare messages
        messages_to_send = []
        
        # If new conversation, add system prompt and greeting
        if is_new_conversation:
            messages_to_send.append(HumanMessage(content=request.message))
        else:
            messages_to_send.append(HumanMessage(content=request.message))
        
        # Invoke agent
        response = agent.invoke(
            {"messages": messages_to_send},
            {"configurable": {"thread_id": request.thread_id}}
        )
        
        # Extract response and summary
        all_messages = response["messages"]
        assistant_response = all_messages[-1].content if all_messages else ""
        
        # Extract summary if available
        summary = extract_summary(all_messages)
        
        # Append summary to response if it exists
        if summary:
            assistant_response = f"{assistant_response}\n\n[Summary: {summary}]"
        
        # Convert conversation history to dict format
        conversation_history = convert_messages_to_dict(all_messages)
        
        return ChatResponse(
            response=assistant_response,
            thread_id=request.thread_id,
            conversation_history=conversation_history,
            conversation_ended=False
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============== Running the app ===============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
