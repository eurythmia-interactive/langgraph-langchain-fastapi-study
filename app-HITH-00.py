from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
import os

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.types import Command

# Load environment variables
load_dotenv()

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

# Initialize Chat Model
llm = ChatOpenAI(
    model=DEFAULT_MODEL,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE,
    temperature=0
)

# Persistent in-memory checkpointer to maintain thread histories and pause states
checkpointer = InMemorySaver()

app = FastAPI(title="Fret & Fiddle Guitar Store - AI Support with HITL")

# =============== Mock Database ===============

MOCK_ORDERS = {
    "1024": {
        "customer": "Sarah Connor",
        "guitar": "Fender American Professional II Stratocaster",
        "price": 1699.99,
        "status": "Delivered"
    },
    "2048": {
        "customer": "John Doe",
        "guitar": "Gibson Les Paul Standard '60s",
        "price": 2799.00,
        "status": "Delivered"
    }
}

# =============== System Prompt ===============

GUITAR_STORE_PROMPT = """You are an AI customer support specialist for 'Fret & Fiddle Guitar Emporium'.
You help customers resolve issues with their instrument purchases.

If a customer complains about cosmetic damage (such as scratches, dents, or finish blemishes) but wants to keep the instrument, follow these steps:
1. Lookup the order status and price using 'check_order_status'.
2. If the order is valid, calculate a partial store credit (strictly up to 10% of the purchase price).
3. Call the 'issue_store_credit' tool with the calculated amount and explanation to initiate the refund process.

Always maintain a professional, friendly, and music-focused tone."""

# =============== Custom State Schema ===============

class GuitarStoreState(AgentState):
    # Standard AgentState (messages list) is sufficient for our current flow.
    pass

# =============== Tools ===============

@tool
def check_order_status(order_id: str) -> str:
    """Lookup the order status and price of an instrument purchase."""
    order = MOCK_ORDERS.get(order_id)
    if not order:
        return f"Order #{order_id} not found."
    return (
        f"Order #{order_id}: {order['guitar']} "
        f"purchased by {order['customer']} for ${order['price']:.2f}. "
        f"Status: {order['status']}."
    )

@tool
def issue_store_credit(order_id: str, amount: float, reason: str) -> str:
    """Issue a store credit refund to the customer's account. This action is sensitive and requires manager approval."""
    # This represents a secure transaction write to your e-commerce database/API
    print(f"\n[CRITICAL TRANSACTION COMMITTED] Issued ${amount:.2f} credit to Order #{order_id}. Reason: {reason}\n")
    return f"Successfully processed store credit of ${amount:.2f} for Order #{order_id}."

# =============== Agent Factory ===============

def create_guitar_agent():
    """Create agent with the HITL middleware intercepting sensitive tool actions"""
    return create_agent(
        model=llm,
        tools=[check_order_status, issue_store_credit],
        checkpointer=checkpointer,
        state_schema=GuitarStoreState,
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={"issue_store_credit": True},  # Force pause before executing this tool
                description_prefix="Manager approval required for store credit"
            )
        ]
    )

# =============== Pydantic API Models ===============

class ChatRequest(BaseModel):
    message: str
    thread_id: str

class ChatResponse(BaseModel):
    response: Optional[str] = None
    thread_id: str
    conversation_history: List[dict]
    is_interrupted: bool = False
    interrupt_details: Optional[dict] = None
    conversation_ended: bool = False

class ManagerDecisionRequest(BaseModel):
    thread_id: str
    decision_type: str  # Must be "approve", "reject", or "edit"
    message: Optional[str] = None  # Rejection reasoning
    edited_action: Optional[Dict[str, Any]] = None  # Struct for edited arguments: {"name": "...", "args": {...}}

# =============== Helper Functions ===============

def convert_messages_to_dict(messages: List) -> List[dict]:
    """Convert LangChain messages into standard JSON-serializable structures"""
    result = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            result.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "assistant", "content": msg.content})
    return result

def extract_interrupt_details(response_data: dict) -> Optional[dict]:
    """Extracts the underlying action requests from LangGraph interrupt payloads"""
    if "__interrupt__" in response_data and response_data["__interrupt__"]:
        try:
            raw_interrupt = response_data["__interrupt__"][0]
            val = getattr(raw_interrupt, "value", raw_interrupt)
            if isinstance(val, dict):
                return {"action_requests": val.get("action_requests", [])}
            return {"raw": str(raw_interrupt)}
        except Exception as e:
            return {"error": f"Failed to parse interrupt metadata: {str(e)}"}
    return None

# =============== API Endpoints ===============

@app.get("/")
async def root():
    return {
        "message": "Fret & Fiddle Guitar Store Support API",
        "endpoints": ["/chat", "/manager/decision"],
        "status": "Online"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for customers.
    Automatically intercepts transactions requiring manager validation.
    """
    try:
        if request.message.strip().lower() == "exit":
            return ChatResponse(
                response="Thank you for visiting Fret & Fiddle! Rock on!",
                thread_id=request.thread_id,
                conversation_history=[],
                conversation_ended=True
            )

        agent = create_guitar_agent()

        # Always inject the system prompt along with the user's message
        messages_to_send = [
            SystemMessage(content=GUITAR_STORE_PROMPT),
            HumanMessage(content=request.message)
        ]

        response = await agent.ainvoke(
            {"messages": messages_to_send},
            {"configurable": {"thread_id": request.thread_id}}
        )

        # Detect if an interrupt occurred
        interrupt_details = extract_interrupt_details(response)
        is_interrupted = interrupt_details is not None

        all_messages = response.get("messages", [])
        assistant_response = all_messages[-1].content if all_messages and not is_interrupted else None
        conversation_history = convert_messages_to_dict(all_messages)

        return ChatResponse(
            response=assistant_response,
            thread_id=request.thread_id,
            conversation_history=conversation_history,
            is_interrupted=is_interrupted,
            interrupt_details=interrupt_details,
            conversation_ended=False
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/manager/decision", response_model=ChatResponse)
async def manager_decision(request: ManagerDecisionRequest):
    """
    Manager endpoint to resolve pending interruptions (Approve, Reject, or Edit).
    """
    try:
        if request.decision_type not in ["approve", "reject", "edit"]:
            raise HTTPException(status_code=400, detail="Invalid decision_type. Use 'approve', 'reject', or 'edit'.")

        agent = create_guitar_agent()

        # Build the decision payload to feed into the state machine
        decision = {"type": request.decision_type}
        if request.decision_type == "reject" and request.message:
            decision["message"] = request.message
        elif request.decision_type == "edit" and request.edited_action:
            decision["edited_action"] = request.edited_action

        resume_command = Command(resume={"decisions": [decision]})

        # Resume execution on the specific thread
        response = await agent.ainvoke(
            resume_command,
            {"configurable": {"thread_id": request.thread_id}}
        )

        # Confirm if any consecutive interrupts arose (usually unlikely in this workflow)
        interrupt_details = extract_interrupt_details(response)
        is_interrupted = interrupt_details is not None

        all_messages = response.get("messages", [])
        assistant_response = all_messages[-1].content if all_messages else ""
        conversation_history = convert_messages_to_dict(all_messages)

        return ChatResponse(
            response=assistant_response,
            thread_id=request.thread_id,
            conversation_history=conversation_history,
            is_interrupted=is_interrupted,
            interrupt_details=interrupt_details,
            conversation_ended=False
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)