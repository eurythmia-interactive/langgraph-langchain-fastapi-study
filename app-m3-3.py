from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.agents import AgentState
from langchain.agents.middleware import after_agent
from langgraph.runtime import Runtime
from typing import Any

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

app = FastAPI(title="AI Engineering Assistant with Multi-Summary")

BASE_SYSTEM_PROMPT = """You are an AI engineering assistant specialized in:
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

# =============== Custom State Schema ===============

class ConversationState(AgentState):
    summaries: list[dict]

# =============== Models ===============

class ChatRequest(BaseModel):
    message: str
    thread_id: str

class ChatResponse(BaseModel):
    response: str
    thread_id: str
    conversation_history: List[dict]
    conversation_ended: bool

class SummariesResponse(BaseModel):
    thread_id: str
    summaries: List[dict]
    count: int

# =============== Summary Generation ===============

async def generate_structured_summary(user_message: str, ai_response: str) -> dict:
    """Generate a structured summary from user message and AI response"""
    prompt = f"""Summarize this exchange in exactly this format (max 50 tokens total):
Topic: [main topic]
Question: [user's question]
Answer: [key points from response]

Exchange:
User: {user_message}
Assistant: {ai_response}"""
    
    response = await llm.ainvoke(prompt, max_tokens=50)
    summary_text = response.content
    
    # Parse the structured response
    summary_dict = {"topic": "", "question": "", "answer": ""}
    
    for line in summary_text.split("\n"):
        line = line.strip()
        if line.startswith("Topic:"):
            summary_dict["topic"] = line.replace("Topic:", "").strip()
        elif line.startswith("Question:"):
            summary_dict["question"] = line.replace("Question:", "").strip()
        elif line.startswith("Answer:"):
            summary_dict["answer"] = line.replace("Answer:", "").strip()
    
    return summary_dict

# =============== Custom Middleware ===============

@after_agent
async def generate_and_store_summary(state: ConversationState, runtime: Runtime) -> dict[str, Any] | None:
    """Generate summary after agent responds and store in state"""
    messages = state["messages"]
    
    if len(messages) < 2:
        return None
    
    # Get last user message and AI response
    last_user_msg = None
    last_ai_msg = None
    
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and last_user_msg is None:
            last_user_msg = msg.content
        elif isinstance(msg, AIMessage) and last_ai_msg is None:
            last_ai_msg = msg.content
        
        if last_user_msg and last_ai_msg:
            break
    
    if not last_user_msg or not last_ai_msg:
        return None
    
    # Skip summary generation if exit command
    if last_user_msg.strip().lower() == "exit":
        return None
    
    # Generate summary
    summary = await generate_structured_summary(last_user_msg, last_ai_msg)
    
    # Get existing summaries or initialize empty list
    existing_summaries = state.get("summaries", [])
    
    # Append new summary
    existing_summaries.append(summary)
    
    # Keep only last 10 summaries
    existing_summaries = existing_summaries[-10:]
    
    return {"summaries": existing_summaries}

# =============== Helper Functions ===============

def format_summaries_for_prompt(summaries: list[dict]) -> str:
    """Format summaries for injection into system prompt"""
    if not summaries:
        return ""
    
    formatted = []
    for i, summary in enumerate(summaries, 1):
        formatted.append(f"{i}. Topic: {summary['topic']}\n   Question: {summary['question']}\n   Answer: {summary['answer']}")
    
    return "\n\n".join(formatted)

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

# =============== Agent Factory ===============

def create_chat_agent_with_summaries():
    """Create agent with custom summary middleware"""
    middleware = [generate_and_store_summary]
    
    return create_agent(
        model=llm,
        checkpointer=checkpointer,
        state_schema=ConversationState,
        middleware=middleware
    )

# =============== Endpoints ===============

@app.get("/")
async def root():
    return {
        "message": "AI Engineering Assistant with Multi-Summary",
        "endpoints": ["/chat", "/summaries/{thread_id}"],
        "description": "Conversational AI assistant with structured summaries"
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
        agent = create_chat_agent_with_summaries()
        
        # Get current state to retrieve summaries
        try:
            state = await agent.aget_state({"configurable": {"thread_id": request.thread_id}})
            summaries = state.values.get("summaries", []) if state and state.values else []
        except:
            summaries = []
        
        # Build system prompt with summaries
        summaries_text = format_summaries_for_prompt(summaries)
        if summaries_text:
            system_prompt = f"{BASE_SYSTEM_PROMPT}\n\nPrevious conversation context:\n{summaries_text}"
        else:
            system_prompt = BASE_SYSTEM_PROMPT
        
        # Prepare messages with system prompt
        messages_to_send = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=request.message)
        ]
        
        # Invoke agent
        response = await agent.ainvoke(
            {"messages": messages_to_send},
            {"configurable": {"thread_id": request.thread_id}}
        )
        
        # Extract response
        all_messages = response["messages"]
        assistant_response = all_messages[-1].content if all_messages else ""
        
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

@app.get("/summaries/{thread_id}", response_model=SummariesResponse)
async def get_summaries(thread_id: str):
    """
    Retrieve all summaries for a conversation thread
    """
    try:
        agent = create_chat_agent_with_summaries()
        
        # Get state for thread
        state = await agent.aget_state({"configurable": {"thread_id": thread_id}})
        
        if not state or not state.values:
            return SummariesResponse(
                thread_id=thread_id,
                summaries=[],
                count=0
            )
        
        summaries = state.values.get("summaries", [])
        
        return SummariesResponse(
            thread_id=thread_id,
            summaries=summaries,
            count=len(summaries)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============== Running the app ===============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
