# LangGraph + FastAPI Study Project

A hands-on learning project exploring LangGraph implementation patterns with FastAPI servers. Each module demonstrates different LangGraph concepts: agents, memory, middleware, human-in-the-loop workflows, and breakpoints.

## Setup

### 1. Create virtual environment
```bash
uv venv
```

### 2. Activate the environment
```bash
source .venv/bin/activate
```

### 3. Install dependencies
```bash
uv pip install -r requirements.txt
```

### 4. Configure environment variables
Create a `.env` file:
```
OPENAI_API_KEY=your-key-here
OPENAI_API_BASE=https://your-proxy-url/v1
DEFAULT_MODEL=your-model-name
```

---

## Modules

### Module 1: Agent Memory with Tools
**File:** `langGraph-agent-memory-00.py`

Demonstrates persistent memory using `MemorySaver` and tool-calling agents with arithmetic operations.

```bash
uv run python langGraph-agent-memory-00.py
```

**Endpoint:** `POST /process`

---

### Module 2: Message Management
**Files:** `M2-LG-*.py`

Covers message manipulation techniques:
- **DELETE Messages** - Removing messages from conversation state
- **Trim Messages** - Trimming conversation history
- **SQLite Memory** - Persistent memory with SQLite checkpointing

---

### Module 3: Middleware & Conversational AI

#### Lesson 1: Message Manager API
**File:** `app-m3-1.py`

Demonstrates middleware patterns with summarization and trimming.

```bash
uv run python app-m3-1.py
uv run python test-m3-1.py  # tests
```

**Endpoints:**
- `POST /process` - Process with optional middleware
- `POST /summarize` - Summarization middleware
- `POST /trim` - Trimming middleware

#### Lesson 2: AI Engineering Assistant
**File:** `app-m3-2.py`

Conversational AI assistant with summarization for context management.

```bash
uv run python app-m3-2.py
```

**Endpoint:** `POST /chat`

```json
{
  "message": "What is RAG?",
  "thread_id": "user-123"
}
```

#### Lesson 3: Multi-Summary Assistant
**File:** `app-m3-3.py`

Enhanced assistant with structured summaries stored per conversation.

```bash
uv run python app-m3-3.py
uv run python test-m3-3.py  # tests
```

**Endpoints:**
- `POST /chat` - Chat with summary generation
- `GET /summaries/{thread_id}` - Retrieve summaries

---

### Module 4: Human-in-the-Loop (HITL)
**File:** `app-HITH-00.py`

Guitar store customer support with manager approval workflow for sensitive actions (store credit issuance).

```bash
uv run python app-HITH-00.py
uv run python test-HITL-00.py  # tests
```

**Endpoints:**
- `POST /chat` - Customer chat (auto-interrupts on credit actions)
- `POST /manager/decision` - Manager approves/rejects/edits

**Flow:**
1. Customer reports issue → Agent checks order
2. Agent calculates credit → Interrupts for approval
3. Manager decides via `/manager/decision`
4. Agent completes or adjusts response

---

### Module 5: Breakpoints
**Files:** `M3-LG-breakingPoints*.py`

Demonstrates LangGraph breakpoint patterns for pausing and resuming graph execution.

---

## Key Concepts Covered

- **StateGraph** - Building custom graph workflows
- **Checkpointers** - Memory persistence (InMemorySaver, SQLite)
- **Middleware** - Before/after agent hooks (summarization, trimming)
- **Tool Calling** - Binding tools to LLMs
- **Human-in-the-Loop** - Interrupts and resume patterns
- **Thread Management** - Conversation isolation via `thread_id`
- **Breakpoints** - Pausing graph execution at specific nodes

---

## Notes

- Each app runs on port 8000 (stop one before starting another)
- Use unique `thread_id` values for conversation isolation
- Checkpointers must be instantiated at module level (not inside request handlers) to persist across requests
- The `DEFAULT_MODEL` from `.env` is used for all LLM calls
